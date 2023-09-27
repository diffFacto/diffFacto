

"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""


import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from difffacto.utils.registry import DECOMPOSERS
from difffacto.utils.misc import checkpoint
from einops import rearrange, repeat

def exists(val):
    return val is not None

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, mask=None):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x, mask), (), False)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv, mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if exists(mask):
            mask = repeat(mask, 'b j -> b h i j', i=n_ctx, h=self.heads)
            max_neg_value = -torch.finfo(weight.dtype).max
            weight.masked_fill_(~mask, max_neg_value)
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None):
        for block in self.resblocks:
            x = block(x, mask=mask)
        return x

@DECOMPOSERS.register_module()
class PartCodeTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        class_cond: bool = True,
        param_cond: bool = True,
        param_dim: int = 6,
        use_mask_in_transformer=False,
    ):
        super().__init__()
        dtype=torch.float32
        self.dtype=torch.float32
        input_channels += int(param_cond) * param_dim + int(class_cond) * n_ctx
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.use_mask_in_transformer=use_mask_in_transformer
        self.class_cond = class_cond
        self.param_cond = param_cond
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        self.aggregate = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.class_embedding = nn.Embedding(n_ctx, n_ctx, device=device, dtype=dtype)
        self.width=width
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, params: List[torch.Tensor], mask: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        if self.param_cond:
            x = torch.cat([x] + params, dim=1)
        class_embed = self.class_embedding(torch.arange(self.n_ctx).to(x.device)).unsqueeze(0).repeat_interleave(x.shape[0], dim=0)
        if self.class_cond:
            x = torch.cat([x, class_embed.transpose(1,2)], dim=1)
        assert self.input_channels == x.shape[1]
        h =  self._forward_with_cond(x, [], mask=mask.to(bool) if self.use_mask_in_transformer else None)
        h = h * mask.unsqueeze(1) # zero out the part code generated from nothing
        z = self.get_global_from_part(h, mask)
        return z, h

    def get_global_from_part(self, h: torch.Tensor, mask: torch.Tensor):
        h = h + (1 - mask.unsqueeze(1)) * -100000
        z = self.aggregate(h.max(2)[0])
        assert z.shape == (h.shape[0], self.output_channels)
        return z
    
    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]], mask: Optional[torch.Tensor]=None
    ) -> List[torch.Tensor]:
        B = x.shape[0]
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h, mask=mask)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        
        
        assert h.shape == (B, self.n_ctx, self.output_channels)
        return h.transpose(1,2)


@DECOMPOSERS.register_module()
class PartCodeTransformerV2(PartCodeTransformer):
    def __init__(self,
        *,
        device: torch.device,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        class_cond: bool = True,
        param_cond: bool = True,
        param_dim: int = 6,
        post_mlp: bool=False
        ):
        assert output_channels % n_ctx == 0
        out_ch_per_part = output_channels // n_ctx
        self.new_out_ch = output_channels
        super().__init__(
            device=device,
            input_channels=input_channels,
            output_channels=out_ch_per_part,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            class_cond=class_cond,
            param_cond=param_cond,
            param_dim=param_dim
        )
        if post_mlp:
            self.post_mlp = MLP(device=device, dtype=self.dtype, width=self.new_out_ch, init_scale=init_scale)
        else:
            self.post_mlp = None
        
    def get_global_from_part(self, h: torch.Tensor, mask: torch.Tensor):
        B, C, N = h.shape
        assert C*N == self.new_out_ch and N == self.n_ctx
        return h.reshape(B, self.new_out_ch) if self.post_mlp is None else self.post_mlp(h.reshape(B, self.new_out_ch))
    