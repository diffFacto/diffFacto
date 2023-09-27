from torch.nn import Module, ModuleList, Linear
import torch
from difffacto.utils.registry import NETS
import torch.nn.functional as F
from difffacto.utils.constants import *
from pointnet2_ops.pointnet2_utils import gather_operation
from .utils import ConcatSquashLinear

@NETS.register_module()
class PointwiseNet2(Module):

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 context_dim, 
                 num_anchors, 
                 res=True, 
                 include_anchors=False, 
                 gather_layers=[1, 3], 
                 add_to_context=False,
                 K=1
                 ):
        super().__init__()
        self.act = F.leaky_relu   
        self.res=res     
        point_dim = 6 if include_anchors else 3
        self.num_anchors = num_anchors
        self.K=K
        self.gather_layers=gather_layers
        out_chs = [in_channels, 128, 256, 512, 256, 128, out_channels]
        self.layers = ModuleList()
        self.add_to_context=add_to_context
        self.context_dim = context_dim + point_dim
        if add_to_context:
            ch = 0
            for i in range(len(out_chs) - 1):
                if i - 1 in gather_layers:
                    ch = out_chs[i]
                self.layers.append(ConcatSquashLinear(out_chs[i], out_chs[i + 1], context_dim + ch + point_dim))
        else:
            for i in range(len(out_chs) - 1):
                self.layers.append(ConcatSquashLinear(out_chs[i] if i - 1 not in gather_layers else out_chs[i] * 2, out_chs[i + 1], context_dim+point_dim))
            
        self.include_anchors=include_anchors
    def gather_idx(self, x, anchor_assignment):
        # compute pointwise distance
        if x.shape[1] > 3:
            x = x[:, :3]
        dist = torch.pow(x.unsqueeze(-1) - x.unsqueeze(-2), exponent=2).sum(1) #[B, N, N]
        print(dist.shape)
        out_id = []
        for i in range(self.num_anchors):
            mask = (anchor_assignment != i).to(torch.float32) * 1e9 
            dist_ = dist + mask.unsqueeze(-1) 
            out_id.append(torch.topk(dist_, k=self.K, dim=1, largest=False)[1]) # [B, K, N]
        return torch.cat(out_id, dim=1) 
    def forward(self, x, beta, context, anchor_assignment, anchors=None, variances=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, d, N).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F) or (B, N, F).
        """
        B, C, N = x.shape
        batch_size, _, num_points = x.size()
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ori_x = x
        out = x.transpose(2, 1)

        if len(context.shape) == 2:
            context = context.view(batch_size, 1, -1)   # (B, 1, F)
            ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        else:
            assert len(context.shape) == 3
            assert context.shape[1] == num_points
            ctx_emb = torch.cat([time_emb.expand(-1, context.shape[1], -1), context], dim=-1)    # (B, N, F+3)
        
        if self.include_anchors:
            ctx_emb = torch.cat([ctx_emb, anchors], dim=-1)
        
        
        for i, layer in enumerate(self.layers):
            # print(ctx_emb.shape)
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1 and i not in self.gather_layers:
                out = self.act(out)
            elif i in self.gather_layers:
                feats = [] #[B, C, 4]
                for j in range(self.num_anchors):
                    ids = anchor_assignment != j
                    mask = ids.to(torch.float32) * -10000
                    feat = torch.max(out * mask.unsqueeze(-1), dim=1)[0]
                    feats.append(feat)
                feats = torch.stack(feats, dim=2).contiguous()
                neighboring_feat = gather_operation(feats, anchor_assignment).transpose(1,2)
                # print(neighboring_feat.shape)
                if self.add_to_context:
                    ctx_emb = ctx_emb[..., :self.context_dim]
                    ctx_emb = torch.cat([ctx_emb, neighboring_feat], dim=-1)
                else:
                    out = torch.cat([out, neighboring_feat], dim=-1)

        return out.transpose(2, 1) + ori_x if self.res else out.transpose(2, 1)