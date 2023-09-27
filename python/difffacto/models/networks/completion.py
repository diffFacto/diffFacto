import torch
from torch.nn import Module
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS, DIFFUSIONS, SAMPLERS

@MODELS.register_module()
class DiffuCompletion(Module):

    def __init__(self, encoder, diffusion, sampler, npoints=2048, ret_traj=False, ret_interval=20):
        super().__init__()
        self.encoder = build_from_cfg(encoder, ENCODERS)
        self.npoints = npoints
        self.diffusion = build_from_cfg(diffusion, DIFFUSIONS)
        self.num_timesteps = self.diffusion.num_timesteps
        self.sampler = build_from_cfg(sampler, SAMPLERS, num_timesteps=self.num_timesteps)
        self.ret_traj=ret_traj
        self.ret_interval=ret_interval

    
    def decode(self, code, noise=None, device='cuda'):
        final = dict() 
        bs = code.shape[0]
        for t, sample in self.diffusion.p_sample_loop_progressive(
            [bs, 3, self.npoints],
            code=code,
            noise=noise,
            device=device,
            progress=False,
        ):
            if t == 0:
                final['pred'] = sample['sample'].transpose(2, 1)
            elif (self.ret_traj and t % self.ret_interval == 0):
                final[t] = sample['sample'].transpose(2, 1)
            
                
        return final
                


    def q_sample(self, input, anchors=None, anchor_assignments=None, ret_traj=False, ret_interval=1, device='cuda'):
        if anchors is None:
            anchors = fps(input, self.num_anchors)
            anchor_assignments = assign_anchor(input, anchors).to(device)
        return self.diffusion.forward_sample(input, anchors, anchor_assignments, ret_traj=ret_traj, ret_interval=ret_interval, device=device)
    
    def forward(self, pcds, device='cuda'):
        fine_gt = pcds['pointcloud'].to(device)
        input = pcds['partial'].to(device)
        code = self.encoder(input)
        if self.training:
            t, _ = self.sampler.sample(input.shape[0], device)
            recons_loss = self.diffusion.training_losses(fine_gt.transpose(2, 1), t, code)
            return recons_loss
        else:
            pred = self.decode(code, device=device)
            pred['ref'] = fine_gt 
            pred['input'] = input
            return pred