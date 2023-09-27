from torch.nn import Module, ModuleList, Linear
from torch import nn, einsum
import torch
from difffacto.utils.registry import NETS
import torch.nn.functional as F
from difffacto.utils.constants import *
from .utils import ConcatSquashLinear, timestep_embedding
from inspect import isfunction 
import math
from einops import rearrange, repeat


class SinusoidalEmbedding3D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim=out_dim 
        if out_dim % 6 :
            self.intermediate_dim = int(math.ceil(out_dim / 6) * 6)
        else:
            self.intermediate_dim = out_dim 
        self.linear = nn.Linear(self.intermediate_dim, out_dim, bias=False)
        self.scaling_factor = torch.ones(self.intermediate_dim // 6, dtype=torch.float32) * (10000 ** (1 / self.intermediate_dim))
        self.scaling_factor = torch.pow(self.scaling_factor, torch.arange(self.intermediate_dim // 6) * 6)

    def forward(self, x):
        # x[B, N, 3]
        B, N, _ = x.shape
        cos_x, sin_x = torch.cos(x.unsqueeze(3) / self.scaling_factor.reshape(1, 1, 1, -1).to(x.device)), torch.sin(x.unsqueeze(3) / self.scaling_factor.reshape(1, 1, 1, -1).to(x.device)) #[B, N, 3, D // 6]
        sinusoidal_pe = torch.stack([sin_x, cos_x], dim=-1).reshape(B, N, self.intermediate_dim) #[B, N, 3, D // 6, 2] -> [B, N, D]
        sinusoidal_pe = self.linear(sinusoidal_pe)
        return sinusoidal_pe

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


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) p -> qkv b heads c p', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


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
        b,c,p = q.shape
        q = rearrange(q, 'b c p -> b p c')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., rel_pe=False, num_part=4, use_anchor_as_pe=False, embed_channel=128, use_scale_shift_norm=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.rel_pe = rel_pe
        if self.rel_pe and not use_anchor_as_pe:
            self.relative_position_bias_table = nn.Parameter(torch.zeros(heads, num_part, num_part))

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

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        if self.rel_pe:
            sim = sim + self.relative_position_bias_table.unsqueeze(0)

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., emb_channels=128, use_scale_shift_norm=True, context_dim=None, gated_ff=True, rel_pe=False, num_part=4, include_anchor_pe=False):
        super().__init__()
        self.use_scale_shift_norm=use_scale_shift_norm
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * dim if use_scale_shift_norm else dim,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(dim, dim, 1)
            ),
        )
        self.include_anchor_pe=include_anchor_pe
        if self.include_anchor_pe:
            self.pe_encoder = SinusoidalEmbedding3D(emb_channels)
            self.pe_emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * dim if use_scale_shift_norm else dim,
                )
            )
            self.pe_out_layers = nn.Sequential(
                Normalize(dim),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    nn.Conv1d(dim, dim, 1)
                ),
            )
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, rel_pe=rel_pe, num_part=num_part)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, rel_pe=rel_pe, num_part=num_part)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, emb, context=None, anchors=None):
        if self.include_anchor_pe:
            pe_embedding = self.emb_layers(self.pe_encoder(anchors.transpose(1,2))).transpose(1,2)
        else:
            pe_embedding = 0.
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        # print(pe_embedding.shape)
        emb_out = emb_out + pe_embedding
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(x) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = x + emb_out
            h = self.out_layers(h)
        x = x + h
        x = x.transpose(1,2)
        if exists(context):
            context = context.transpose(1,2)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        x = x.transpose(1,2)
        return x

@NETS.register_module()
class UNet(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, embed_dim=128, use_scale_shift_norm=True, dropout=0., prior_dim=512, language_dim=64, gated_ff=True, rel_pe=False, num_part=4, include_anchor_pe=False):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim=embed_dim
        inner_dim = n_heads * d_head
        self.in_layers = nn.Sequential(
            Normalize(in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, inner_dim, 1),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.transformer_blocks_prior = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=prior_dim, use_scale_shift_norm=use_scale_shift_norm, gated_ff=gated_ff, emb_channels=embed_dim, rel_pe=rel_pe, num_part=num_part, include_anchor_pe=include_anchor_pe)
                for d in range(depth)]
        )
        self.transformer_blocks_language = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=language_dim, use_scale_shift_norm=use_scale_shift_norm, gated_ff=gated_ff, emb_channels=embed_dim, rel_pe=rel_pe, num_part=num_part, include_anchor_pe=include_anchor_pe)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                              in_channels, 1))

    def forward(self, x, t, prior=None, language_code=None, part_indicator=None, anchors=None):
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        x = self.in_layers(x)
        t_emb = timestep_embedding(t, self.embed_dim, repeat_only=False)
        emb = self.time_embed(t_emb)
        for i in range(len(self.transformer_blocks_prior)):
            x = self.transformer_blocks_prior[i](x, emb, context=prior, anchors=anchors)
            x = self.transformer_blocks_language[i](x, emb, context=language_code, anchors=anchors)
        x = self.proj_out(x)
        return x + x_in 