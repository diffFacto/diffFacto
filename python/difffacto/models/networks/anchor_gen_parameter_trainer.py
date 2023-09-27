from torch.nn import Module
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS

@MODELS.register_module()
class ParameterTrainer(Module):

    def __init__(
        self, 
        encoder, 
        num_anchors,
        npoints=2048, 


        ):
        super().__init__()
        self.encoder = build_from_cfg(encoder, ENCODERS)
        self.num_anchors = num_anchors
        
        self.npoints = npoints
        self.points_per_anchor = self.npoints // self.num_anchors

        print(self)


    def forward(self, pcds, device='cuda', **kwargs):
        """
            input: [B, npoints, 3]
            attn_map: [B, npoints, n_class]
            seg_mask: [B, npoints]
        """


        ctx, mean_per_point, logvar_per_point, flag_per_point, fit_los_dict = self.encoder(pcds, device)
        
        return fit_los_dict
