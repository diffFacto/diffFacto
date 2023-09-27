from torch.nn import Module, ModuleList, Linear
import torch
from difffacto.utils.registry import NETS
import torch.nn.functional as F
from difffacto.utils.constants import *
from .utils import ConcatSquashLinear

@NETS.register_module()
class PointwiseNetLatent(Module):

    def __init__(self, in_channels, out_channels, context_dim, res=True, use_part_ind=False, **kwargs):
        super().__init__()
        self.act = F.leaky_relu   
        self.res=res     

        self.layers = ModuleList([
            ConcatSquashLinear(in_channels, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, out_channels, context_dim+3)
        ])
        self.use_part_ind=use_part_ind
    def forward(self, x, beta, prior, code, part_indicator, anchors=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, d, N).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F) or (B, N, F).
        """
        batch_size, _, num_points = x.size()
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ori_x = x
        out = x.transpose(2, 1)
        prior = prior.transpose(1, 2)
        
        if len(code.shape) == 2:
            code = code.unsqueeze(1)
        elif len(code.shape) == 3:
            code = code.transpose(1,2)
        # print(code.shape, prior.shape)
        code = code.expand(-1, prior.shape[1], -1)
        if self.use_part_ind:
            code = code * part_indicator.unsqueeze(-1) # zero out the language latent for unrelavent parts
        ctx_emb = torch.cat([time_emb.expand(-1, prior.shape[1], -1), prior, code], dim=-1)
        assert len(ctx_emb.shape) == 3
        assert ctx_emb.shape[1] == num_points
        
        
        
        
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        return out.transpose(2, 1) + ori_x if self.res else out.transpose(2, 1)
    
@NETS.register_module()
class PointwiseNet(Module):

    def __init__(self, in_channels, out_channels, context_dim, res=True):
        super().__init__()
        self.act = F.leaky_relu   
        self.res=res     
        point_dim = 3 


        self.layers = ModuleList([
            ConcatSquashLinear(in_channels, 128, context_dim+point_dim),
            ConcatSquashLinear(128, 256, context_dim+point_dim),
            ConcatSquashLinear(256, 512, context_dim+point_dim),
            ConcatSquashLinear(512, 256, context_dim+point_dim),
            ConcatSquashLinear(256, 128, context_dim+point_dim),
            ConcatSquashLinear(128, out_channels, context_dim+point_dim)
        ])
    def forward(self, x, beta, context, **kwargs):
        """
        Args:
            x:  Point clouds at some timestep t, (B, d, N).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F) or (B, N, F).
        """
        batch_size, _, num_points = x.size()
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ori_x = x
        out = x.transpose(2, 1)
        if(context[0].shape[1] == 1):
            ctx_emb = torch.cat([time_emb] + context, dim=-1)    # (B, N, F+3)
        else:
            ctx_emb = torch.cat([time_emb.repeat_interleave(num_points, dim=1)] + context, dim=-1)    # (B, N, F+3)
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        return out.transpose(2, 1) + ori_x if self.res else out.transpose(2, 1)