import torch 
from torch import nn 
import numpy as np
from difffacto.utils.registry import DECOMPOSERS 
import torch.nn.functional as F
from .common import GAT, MultiHeadSelfAttention, SinusoidalEmbedding3D
from einops import rearrange, einsum, repeat

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum(q, k, 'b i d, b j d -> b i j') * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum(attn, v, 'b i j, b j d -> b i d')
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

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
        dim_out = dim
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

def identity(*args, **kwargs):
    return nn.Identity()
@DECOMPOSERS.register_module()
class ComponentMixer(nn.Module):
    def __init__(
        self, 
        num_anchors, 
        part_latent_dim, 
        point_dim=3, 
        include_attention=False, 
        nheads=8, 
        use_graph_attention=True, 
        use_abs_pe=False,
        include_global_feature=False,
        global_mlp_type=0,
        normalize_latent=False,
        deprecation=False,
        mlp_type=0,
        norm=None,
        mlp_norm=-1, 
        regressor_norm=-1,
        embed_channel=128,
        use_scale_shift_norm=False,
        pe_dp = 0.2,
        pe_norm=None,
        res=True,
        attn_ln=True,
    ):
        super().__init__()
        self.num_anchors=num_anchors 
        self.global_mlp_type=global_mlp_type
        self.point_dim = point_dim
        self.include_attention=include_attention
        self.use_abs_pe=use_abs_pe
        self.include_global_feature=include_global_feature
        self.normalize_latent=normalize_latent
        self.mlp_type=mlp_type
        if mlp_norm == -1 and regressor_norm == -1:
            if norm == 'bn':
                mlp_norm = nn.BatchNorm1d
                regressor_norm = nn.BatchNorm1d 
            elif norm == 'gn':
                mlp_norm = Normalize
                regressor_norm = Normalize 
            elif norm is None:
                mlp_norm = identity
                regressor_norm = identity
        else:
            if mlp_norm == 'bn':
                mlp_norm = nn.BatchNorm1d
            elif mlp_norm == 'gn':
                mlp_norm = Normalize
            elif mlp_norm is None:
                mlp_norm = identity
            if regressor_norm == 'bn':
                regressor_norm = nn.BatchNorm1d
            elif regressor_norm == 'gn':
                regressor_norm = Normalize
            elif regressor_norm is None:
                regressor_norm = identity
                
        if mlp_type == 1:
            self.mlp = nn.Sequential(
                mlp_norm(part_latent_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(part_latent_dim, 256, 1),
                mlp_norm(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, part_latent_dim, 1),
            )
            self.anchor_regressor = nn.Sequential(
                regressor_norm(part_latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(part_latent_dim, 128 ),
                regressor_norm(128),
                nn.ReLU(inplace=True),
                nn.Linear(128 , point_dim * num_anchors)
            )
        elif mlp_type == 0:
            self.mlp = nn.Sequential(
                nn.Linear(part_latent_dim * num_anchors, 1024),
                mlp_norm(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                mlp_norm(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, point_dim * num_anchors)
            )
        self.deprecation=deprecation
        if deprecation:
            self.mlp_type=-1
            print("THIS VERSION IS DEPRECATED USE MLP TYPE INSTEAD")
            self.mlp = nn.Sequential(
                nn.Linear(part_latent_dim * num_anchors, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, point_dim * num_anchors)
            )
            if self.use_abs_pe:
                self.pe_encoder = SinusoidalEmbedding3D(part_latent_dim)
        
        if self.include_attention:
            self.res=res
            self.use_scale_shift_norm=use_scale_shift_norm
            if pe_norm == 'bn':
                pe_norm = nn.BatchNorm1d
            elif pe_norm == 'gn':
                pe_norm = Normalize
            elif pe_norm is None:
                pe_norm = identity
            if self.use_abs_pe:
                self.pe_encoder = SinusoidalEmbedding3D(embed_channel)
                self.emb_layers = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(
                        embed_channel,
                        2 * part_latent_dim if use_scale_shift_norm else part_latent_dim,
                    )
                )
                self.out_layers = nn.Sequential(
                    pe_norm(part_latent_dim),
                    nn.SiLU(),
                    nn.Dropout(p=pe_dp),
                    zero_module(
                        nn.Conv1d(part_latent_dim, part_latent_dim, 1)
                    ) if self.res else nn.Conv1d(part_latent_dim, part_latent_dim, 1),
                )

            self.attention = GAT(part_latent_dim, part_latent_dim, nheads) if use_graph_attention else MultiHeadSelfAttention(part_latent_dim, part_latent_dim, nheads=nheads)
            self.ff = FeedForward(part_latent_dim, part_latent_dim, glu=True)
            self.norm1 = nn.LayerNorm(part_latent_dim) if attn_ln else nn.Identity()
            self.norm2 = nn.LayerNorm(part_latent_dim) if attn_ln else nn.Identity()

        if self.include_global_feature:
            if self.global_mlp_type == 0:
                self.aggregator = nn.Sequential(
                    nn.Linear(part_latent_dim * num_anchors, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, part_latent_dim)
                )
            elif self.global_mlp_type == 1:
                self.aggregator = nn.Sequential(
                    nn.BatchNorm1d(part_latent_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(part_latent_dim, 256, 1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(256, part_latent_dim, 1),
                )
                
                self.aggregator_mlp = nn.Sequential(
                    nn.Linear(part_latent_dim, 256),
                    nn.BatchNorm1d(part_latent_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, part_latent_dim, 1),
                )
                

    def forward(self, part_latent):
        B = part_latent.shape[0]
        if self.mlp_type == 0:
            coarse = self.mlp(part_latent.reshape(B, -1)).reshape(B, self.num_anchors, self.point_dim)
        elif self.mlp_type == 1:
            coarse_feat = self.mlp(part_latent.reshape(B, self.num_anchors, -1).transpose(1,2)).max(-1)[0]
            coarse = self.anchor_regressor(coarse_feat).reshape(B, self.num_anchors, self.point_dim)

        if self.deprecation:
            coarse = self.mlp(part_latent.reshape(B, -1)).reshape(B, self.num_anchors, self.point_dim)
            if self.use_abs_pe:
                pe_embedding = self.pe_encoder(coarse)
                part_latent = part_latent + pe_embedding.detach()

        if self.include_attention:
            if self.use_abs_pe:
                pe_embedding = self.emb_layers(self.pe_encoder(coarse)).transpose(1,2)
                _part_latent = part_latent.transpose(1,2)
                if self.use_scale_shift_norm:
                    emb_norm, emb_rest = self.out_layers[0], self.out_layers[1:]
                    scale, shift = torch.chunk(pe_embedding, 2, dim=1)
                    part_latent_out = emb_norm(_part_latent) * (1 + scale) + shift
                    part_latent_out = emb_rest(part_latent_out).transpose(1,2)
                else:
                    part_latent_out = _part_latent + pe_embedding
                    part_latent_out = self.out_layers(part_latent_out).transpose(1,2)
                
                part_latent = part_latent + part_latent_out if self.res else part_latent_out
            part_latent = self.attention(self.norm1(part_latent)) + part_latent if self.res else self.attention(self.norm1(part_latent))
            part_latent = self.ff(self.norm2(part_latent)) + part_latent if self.res else self.ff(self.norm2(part_latent))
            
        if self.include_global_feature:
            if self.global_mlp_type == 0:
                global_feature = self.aggregator(part_latent.reshape(B, -1))
            elif self.global_mlp_type == 1:
                global_feature = self.aggregator(part_latent.transpose(1,2))
                global_feature = torch.max(global_feature, dim=-1)[0]
                global_feature = self.aggregator_mlp(global_feature)
        else:
            global_feature = None
            
        if self.normalize_latent:
            global_feature = F.normalize(global_feature)
            part_latent = F.normalize(part_latent.reshape(B* self.num_anchors, -1)).reshape(B, self.num_anchors, -1)
        
        
        return global_feature, coarse, part_latent




@DECOMPOSERS.register_module()
class ComponentMixerV2(nn.Module):
    def __init__(
        self, 
        num_anchors, 
        part_latent_dim,
        embed_channel, 
        use_scale_shift_norm=True,
        point_dim=3, 
        include_attention=False, 
        nheads=8, 
        use_graph_attention=True, 
        include_global_feature=False,
        dropout=0.2,
        normalize_latent=False,
        correct_anchors=True,
        mult=1,
        part_dp_prob=0.,
        global_dp_prob=0.
    ):
        super().__init__()
        self.num_anchors=num_anchors 
        self.correct_anchors=correct_anchors
        self.point_dim = point_dim
        self.include_attention=include_attention
        self.include_global_feature=include_global_feature
        self.normalize_latent=normalize_latent
        self.use_scale_shift_norm=use_scale_shift_norm
        self.part_latent_dropout = nn.Dropout(part_dp_prob)
        self.global_latent_dropout = nn.Dropout(global_dp_prob)

        
        self.mlp = nn.Sequential(
            Normalize(part_latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(part_latent_dim, 128 * mult, 1),
            Normalize(128 * mult),
            nn.ReLU(inplace=True),
            nn.Conv1d(128 * mult, part_latent_dim, 1),
        )
        self.anchor_regressor = nn.Sequential(
            Normalize(part_latent_dim* mult),
            nn.ReLU(inplace=True),
            nn.Linear(part_latent_dim, 128 * mult),
            Normalize(part_latent_dim* mult),
            nn.ReLU(inplace=True),
            nn.Linear(128 * mult, point_dim * num_anchors)
        )

        self.pe_encoder = SinusoidalEmbedding3D(embed_channel)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                embed_channel,
                2 * part_latent_dim if use_scale_shift_norm else part_latent_dim,
            )
        )
        self.out_layers = nn.Sequential(
            Normalize(part_latent_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(part_latent_dim, part_latent_dim, 1)
            ),
        )

        if self.include_attention:
            self.attention=GAT(part_latent_dim, part_latent_dim, nheads) if use_graph_attention else MultiHeadSelfAttention(part_latent_dim, part_latent_dim, nheads=nheads)
            self.ff = FeedForward(part_latent_dim, part_latent_dim, glu=True)
            self.norm1 = nn.LayerNorm(part_latent_dim)
            self.norm2 = nn.LayerNorm(part_latent_dim)

        if self.include_global_feature:
            self.aggregator = nn.Sequential(
                nn.Conv1d(part_latent_dim * 2, 256 * mult, 1),
                Normalize(256 * mult),
                nn.ReLU(inplace=True),
                nn.Conv1d(256 * mult, 256 * mult, 1),
                Normalize(256 * mult),
                nn.ReLU(inplace=True),
                nn.Conv1d(256 * mult, part_latent_dim, 1)
            )
        if self.correct_anchors:
            self.anchor_corrector = CrossAttention(self.point_dim, part_latent_dim, 8, 32)

    def forward(self, part_latent):
        B = part_latent.shape[0]
        part_latent = part_latent.reshape(B, self.num_anchors, -1).transpose(1,2)
        coarse_feat = self.mlp(part_latent).max(-1)[0]
        coarse = self.anchor_regressor(coarse_feat).reshape(B, self.num_anchors, self.point_dim)

        pe_embedding = self.emb_layers(self.pe_encoder(coarse)).transpose(1,2)
        if self.use_scale_shift_norm:
            emb_norm, emb_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(pe_embedding, 2, dim=1)
            part_latent_out = emb_norm(part_latent) * (1 + scale) + shift
            part_latent_out = emb_rest(part_latent_out)
        else:
            part_latent_out = part_latent + pe_embedding
            part_latent_out = self.out_layers(part_latent_out)
        
        part_latent = part_latent + part_latent_out
        part_latent = part_latent.transpose(1,2)
        if self.include_attention:
            part_latent = self.attention(self.norm1(part_latent)) + part_latent
            part_latent = self.ff(self.norm2(part_latent)) + part_latent
        part_latent = part_latent.transpose(1,2)
        if self.include_global_feature:
            global_feature = self.aggregator(torch.cat([part_latent, coarse_feat.unsqueeze(2).expand(-1, -1, self.num_anchors)], dim=1)).max(2)[0]
            global_feature = self.global_latent_dropout(global_feature)
        else:
            global_feature = None
        part_latent = part_latent.transpose(1,2)
        part_latent = self.part_latent_dropout(part_latent)
        if self.correct_anchors:
            coarse = coarse + self.anchor_corrector(coarse, part_latent)
        if self.normalize_latent:
            global_feature = F.normalize(global_feature)
            part_latent = F.normalize(part_latent.reshape(B* self.num_anchors, -1)).reshape(B, self.num_anchors, -1)
        return global_feature, coarse, part_latent

