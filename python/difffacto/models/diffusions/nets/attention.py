from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any, Union, List, Tuple

from difffacto.utils.misc import checkpoint
from .unet import timestep_embedding
from .utils import SineLayer
from difffacto.utils.registry import NETS


try:
    import xformers
    import xformers.ops
    print(torch.cuda.get_device_name(0))
    XFORMERS_IS_AVAILBLE = (torch.cuda.get_device_name(0) not in  ['NVIDIA TITAN RTX', 'TITAN Xp', 'NVIDIA TITAN Xp', 'Tesla V100-DGXS-16GB', 'NVIDIA TITAN X (Pascal)'])
except:
    XFORMERS_IS_AVAILBLE = False    

print(XFORMERS_IS_AVAILBLE)
def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class MLP(nn.Module):

    def forward(self, x, *_):
        return self.net(x)

    def __init__(self, ch: Union[List[int], Tuple[int, ...]], use_linear: bool = False, act: nn.Module = nn.ReLU,
                 weight_norm=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.Linear(ch[i], ch[i + 1]) if use_linear else nn.Conv1d(ch[i], ch[i + 1], 1))
            if weight_norm:
                layers[-1] = nn.utils.weight_norm(layers[-1])
            if i < len(ch) - 2:
                layers.append(act(True))
        self.net = nn.Sequential(*layers)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c n -> b n c')
        k = rearrange(k, 'b c n -> b c n')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k)* self.scale
        del q, k

        if exists(mask):
            assert mask.shape == (x.shape[0], context.shape[1])
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h).to(bool)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if exists(mask):
            assert mask.shape == (b, context.shape[1])
            max_neg_value = -torch.finfo(x.dtype).max 
            mask = (1 - mask) * max_neg_value
            mask = mask.repeat_interleave(self.heads, dim=0).view(b*self.heads, 1, context.shape[1]).expand(-1, x.shape[1], -1)
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False,
                 single_attn=False, adaln=False, y_dim=32):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.single_attn = single_attn
        self.adaln=adaln
        
        if not self.single_attn:
            self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                context_dim=context_dim)  # is a self-attention if not self.disable_self_attn
            self.norm1 = nn.LayerNorm(dim)
                
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        if adaln:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    y_dim,
                    2 * dim,
                ),
            )
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, y=None):
        return checkpoint(self._forward, (x, context, mask, y), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None, y=None):
        if not self.single_attn:
            x = self.attn1(self.norm1(x), context=context , mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        if self.adaln:
            shift, scale = torch.chunk(self.emb_layers(y), 2, dim=-1)
            _x = self.norm3(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            _x = self.norm3(x)
        x = self.ff(_x) + x
        return x

@NETS.register_module()
class TransformerNet(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head, out_channels,
                 depth=1, dropout=0., context_dim=None, use_linear=False,
                 use_checkpoint=False, single_attn=False, class_cond=False, n_class=4,
                 cat_params_to_x=False, mask_out_unreferenced_code=True, cat_class_to_x=False, 
                 use_sine_proj_in=False, add_t_to_x=False, res=False, add_class_cond=False,
                 context_proj=False, include_std=False
                ):
        super().__init__()
        self.class_cond=class_cond
        self.add_class_cond=add_class_cond
        self.cat_class_to_x=cat_class_to_x
        in_channels = in_channels + int(cat_params_to_x) * 6 + int(cat_class_to_x) * n_class
        self.in_channels = in_channels
        self.mask_out_unreferenced_code=mask_out_unreferenced_code
        inner_dim = n_heads * d_head
        self.inner_dim = inner_dim
        self.context_dim =  context_dim + 256 + int(class_cond and not add_class_cond) * n_class if not add_t_to_x else context_dim + int(class_cond and not add_class_cond) * n_class
        if class_cond and add_class_cond:
            self.class_emb = nn.Embedding(n_class, inner_dim)
        self.n_class=n_class
        self.include_std=include_std
        self.cat_params_to_x=cat_params_to_x
        self.add_t_to_x=add_t_to_x
        self.res=res
        self.context_proj=context_proj
        if context_proj:
            self.ctx_proj_in = nn.Linear(self.context_dim,inner_dim)
            self.ctx_norm = nn.LayerNorm(inner_dim)
        self.pre_norm = nn.LayerNorm(inner_dim)
        self.post_norm = nn.LayerNorm(inner_dim)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0) if not use_sine_proj_in else SineLayer(in_channels, inner_dim, bias=True, is_first=True, use_linear=True)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim) if not use_sine_proj_in else SineLayer(in_channels, inner_dim, bias=True, is_first=True, use_linear=False)

        self.time_embed = FeedForward(256, dropout=dropout, glu=True) if not add_t_to_x else FeedForward(inner_dim, dropout=dropout, glu=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=self.context_dim if not self.context_proj else inner_dim,
                                   checkpoint=use_checkpoint, single_attn=single_attn, adaln=False)
                for d in range(depth)]
        )
        if not use_linear:
            if use_sine_proj_in:
                self.proj_out = MLP([inner_dim] + [inner_dim * 2] * 5 + [out_channels], use_linear=False)
            else:
                self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)) \
                                            if in_channels == out_channels or res else \
                                            nn.Conv1d(inner_dim,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
        else:
            if use_sine_proj_in:
                self.proj_out = MLP([inner_dim] + [inner_dim * 2] * 5 + [out_channels], use_linear=True)
            else:
                self.proj_out = zero_module(nn.Linear(inner_dim, out_channels)) if in_channels == out_channels or res else nn.Linear(inner_dim, out_channels)
        self.use_linear = use_linear

    def forward(self, x, t, ctx, anchors=None, variances=None, valid_id=None, anchor_assignment=None, **kwargs):
        if isinstance(ctx, list):
            ctx = torch.cat(ctx, dim=1)
        ctx = rearrange(ctx, 'b c n -> b n c').contiguous()
        
        if self.class_cond and not self.add_class_cond:
            class_embed = torch.eye(self.n_class).to(x).unsqueeze(0).repeat_interleave(x.shape[0], dim=0) # B, n_ctx, n_ctx
            ctx = torch.cat([ctx, class_embed], dim=-1)
        t_embed = self.time_embed(timestep_embedding(t, 256 if not self.add_t_to_x else self.inner_dim))
        if not self.add_t_to_x:
            ctx = torch.cat([ctx, t_embed.unsqueeze(1).expand(-1, ctx.shape[1], -1)], dim=-1)
        
        assert ctx.shape[-1] == self.context_dim
        if self.cat_params_to_x:
            if not self.include_std:
                x = torch.cat([x, anchors.transpose(1,2), variances.transpose(1,2)], dim=1)
            else:
                x = torch.cat([x, anchors.transpose(1,2), torch.sqrt(variances).transpose(1,2)], dim=1)
                
            
        if self.cat_class_to_x:
            cls_onehot = F.one_hot(anchor_assignment.to(torch.long), num_classes=self.n_class).transpose(1,2)
            x = torch.cat([x, cls_onehot], dim=1)
        assert x.shape[1] == self.in_channels
        return self._forward_attn(x, ctx, mask=valid_id if self.mask_out_unreferenced_code else None, t_embed=t_embed)
    
    def _forward_attn(self, x, context=None, mask=None, t_embed=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, n = x.shape
        x_in = x
        if self.context_proj:
            context = self.ctx_proj_in(context)
            if self.add_class_cond:
                context = context + self.class_emb(torch.arange(self.n_class, device=x.device)).unsqueeze(0)
            context = self.ctx_norm(context)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c n -> b n c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        if self.add_t_to_x:
            x = x + t_embed.unsqueeze(1)
        x = self.pre_norm(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, mask=mask)
        x = self.post_norm(x)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b n c -> b c n').contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        if x_in.shape[1] == x.shape[1]:
            return x + x_in
        if self.res:
            return x + x_in[:, :x.shape[1]]
        return x
    

@NETS.register_module()
class LDMNet(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head, out_channels,
                 depth=1, dropout=0., context_dim=None, use_linear=False,
                 use_checkpoint=False, single_attn=False, class_cond=False, n_class=4,
                 mask_out_unreferenced_code=True,  cond_time_as_token=True
                ):
        super().__init__()
        self.class_cond=class_cond
        in_channels = in_channels
        self.in_channels = in_channels
        self.mask_out_unreferenced_code=mask_out_unreferenced_code
        inner_dim = n_heads * d_head
        self.inner_dim = inner_dim
        self.context_dim =  context_dim
        if class_cond:
            self.class_emb = nn.Embedding(n_class, inner_dim)
        self.n_class=n_class
        self.cond_time_as_token=cond_time_as_token
        self.pre_norm = nn.LayerNorm(inner_dim)
        self.post_norm = nn.LayerNorm(inner_dim)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0) 
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim) 

        self.time_embed = FeedForward(inner_dim, dropout=dropout, glu=True) 
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=self.context_dim,
                                   checkpoint=use_checkpoint, single_attn=single_attn)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)) 
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels))
        self.use_linear = use_linear

    def forward(self, x, t, ctx, valid_id=None,  **kwargs):
        if isinstance(ctx, list):
            ctx = torch.cat(ctx, dim=1)
        if ctx is not None:
            ctx = rearrange(ctx, 'b c n -> b n c').contiguous()
        
        if self.class_cond:
            class_embed = self.class_emb(torch.arange(self.n_class).to(x.device)).unsqueeze(0).repeat_interleave(x.shape[0], dim=0) # B, n_ctx, n_ctx
        t_embed = self.time_embed(timestep_embedding(t, 256))
        
                
            
        assert x.shape[1] == self.in_channels
        return self._forward_with_cond(x, cond_as_token=[(t_embed, self.cond_time_as_token), (class_embed, False)] if self.class_cond else [(t_embed, self.cond_time_as_token)], ctx=ctx, mask=valid_id if self.mask_out_unreferenced_code else None)
    
    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]], ctx: torch.Tensor=None, mask: torch.Tensor=None
    ) -> torch.Tensor:
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c n -> b n c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for emb, as_token in cond_as_token:
            if not as_token:
                x = x + emb[:, None] if len(emb.shape) == 2 else emb
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            x = torch.cat(extra_tokens + [x], dim=1)
            if mask is not None:
                new_mask = torch.ones(x.shape[0], len(extra_tokens) + self.n_class).to(x)
                new_mask[:, len(extra_tokens):] = mask
                mask = new_mask

        x = self.pre_norm(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=ctx, mask=mask)
        x = self.post_norm(x)
        if len(extra_tokens):
            x = x[:, sum(h.shape[1] for h in extra_tokens) :]
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b n c -> b c n').contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x