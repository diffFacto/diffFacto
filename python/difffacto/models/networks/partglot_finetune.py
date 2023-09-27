import torch
from torch.nn import Module, functional
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS, DIFFUSIONS, SAMPLERS, METRICS, DECOMPOSERS, SEGMENTORS
from pointnet2_ops.pointnet2_utils import gather_operation

@MODELS.register_module()
class PartglotFinetune(Module):

    def __init__(
        self, 
        encoder, 
        diffusion, 
        sampler, 
        num_anchors,
        num_timesteps, 
        npoints=2048, 
        ):
        super().__init__()
        self.sup_segs_encoder = build_from_cfg(encoder, ENCODERS, num_anchors=num_anchors)
        self.diffusion = build_from_cfg(diffusion, DIFFUSIONS, num_timesteps=num_timesteps, learn_variance=False)
        self.sampler = build_from_cfg(sampler, SAMPLERS, num_timesteps=num_timesteps)
        self.num_timesteps = int(num_timesteps)
        self.num_anchors = num_anchors
        self.npoints = npoints
        self.points_per_anchor = self.npoints // self.num_anchors
    


    def forward(self, pcds, device='cuda'):
        pass