from distutils.command.build import build
import torch
import torch.nn as nn 
from torch.nn import Module, functional
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS, DIFFUSIONS, SAMPLERS, METRICS, DECOMPOSERS, SEGMENTORS
from pointnet2_ops.pointnet2_utils import gather_operation
from .language_utils.language_util import tokenizing
VOCAB_SIZE=2787

@MODELS.register_module()
class AnchoredGeneration(Module):

    def __init__(
        self, 
        encoder, 
        decomposer,
        parameter_regressor,
        diffusion, 
        sampler, 
        num_anchors,
        num_timesteps, 
        npoints=2048, 
        anchor_loss_weight=1.,
        loss=None,
        completion=False,
        include_attn_weight_in_encoder=True,
        guidance=False,
        part_latent_dropout_prob=0.2,
        global_latent_dropout_prob=0.2,
        kl_weight=1e-3,
        ret_traj=False, 
        ret_interval=20, 
        forward_sample=False,
        interpolate=False,
        combine=False,
        save_pred_xstart=False,
        fix_part=False
        ):
        super().__init__()
        self.encoder = build_from_cfg(encoder, ENCODERS, num_anchors=num_anchors)
        self.diffusion = build_from_cfg(diffusion, DIFFUSIONS, num_timesteps=num_timesteps, learn_variance=False)
        self.decomposer = build_from_cfg(decomposer, DECOMPOSERS, num_anchors=num_anchors, point_dim=3)
        self.parameter_regressor = build_from_cfg(parameter_regressor, ENCODERS, num_class=num_anchors)
        self.sampler = build_from_cfg(sampler, SAMPLERS, num_timesteps=num_timesteps)
        self.loss_func=build_from_cfg(loss, METRICS)
        self.latent_dim = self.encoder.part_latent_dim
        self.fix_part=fix_part
        self.kl_weight=kl_weight
        self.completion = completion
        self.include_attn_weight_in_encoder=include_attn_weight_in_encoder
        self.num_timesteps = int(num_timesteps)
        self.num_anchors = num_anchors
        self.npoints = npoints
        self.points_per_anchor = self.npoints // self.num_anchors
        self.forward_sample = forward_sample
        self.anchor_loss_weight=anchor_loss_weight
        self.interpolate=interpolate
        self.ret_traj=ret_traj
        self.ret_interval=ret_interval
        self.save_pred_xstart=save_pred_xstart
        self.combine=combine
        self.guidance=guidance
        self.part_latent_dropout_prob = part_latent_dropout_prob
        self.global_latent_dropout_prob=global_latent_dropout_prob
    
    def decode(self, anchors, code=None, pointwise_latent=None, noise=None, variance=None, device='cuda'):
        final = dict() 
        bs = anchors.shape[0]
        for t, sample in self.diffusion.p_sample_loop_progressive(
            [bs, 3, self.npoints],
            anchors=anchors,
            code=code,
            variance=variance,
            pointwise_latent=pointwise_latent,
            noise=noise,
            device=device,
            progress=False,
        ):
            if t == 0:
                final['pred'] = sample['sample'].transpose(2, 1)
                final['pred_xstart'] = sample['pred_xstart'].transpose(1,2) 
            elif (self.ret_traj and t % self.ret_interval == 0):
                final[t] = sample['sample'].transpose(2, 1) 
                final[f'pred_xstart_{t}'] = sample['pred_xstart'].transpose(1,2) 
                
        return final

    def q_sample(self, gt, anchors, noise=None, device='cuda',variance=None):
        final = {'pred':gt.transpose(2,1)}
        
        for t, sample in self.diffusion.q_sample_loop_progressive(
            gt_pcd=gt,
            anchors=anchors,
            noise=noise,
            device=device,
            variance=variance,
            progress=False,
        ):
            if t == self.num_timesteps - 1:
                final[t] = sample.transpose(2, 1)
            elif (self.ret_traj and t % self.ret_interval == 0):
                final[t] = sample.transpose(2, 1)
                
        return final

    def interpolate_latent(self, x, seg_mask, id1, id2, anchor_id, device, pcds):
        gt1, gt2 = x[id1], x[id2]
        latent1 = self.encoder(xyz=gt1.unsqueeze(0))
        latent2 = self.encoder(xyz=gt2.unsqueeze(0))
        latent1, latent2 = latent1.squeeze(0), latent2.squeeze(0) # [1, n_class, C]

        dx = torch.zeros([10, self.num_anchors, 1]).to(device)
        dx[:, anchor_id] += torch.linspace(0, 1, steps=10).reshape(-1, 1).to(device)
        latents = latent1.unsqueeze(0) + (latent2 - latent1).unsqueeze(0) * dx
        global_feat, new_params, new_latents = self.decomposer(latents)
        latent_per_point = new_latents.reshape(10, self.num_anchors, 1, -1).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, -1)
        param_per_point = new_params.reshape(10, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, 3).transpose(1,2)
        assert param_per_point.shape[1] == 3
        anchor_per_point = param_per_point
        variance_per_point = None


        pred = self.decode(anchor_per_point, code=global_feat, pointwise_latent=latent_per_point, device=device, variance=variance_per_point) 
        ref = torch.cat([gt1[seg_mask[id1] != anchor_id, :3], gt2[seg_mask[id2] == anchor_id, :3]], dim=0)
        seg_mask_ref = torch.cat([seg_mask[id1][seg_mask[id1] != anchor_id], seg_mask[id2][seg_mask[id2] == anchor_id]], dim=0)
        pred['input_ref'] = ref.unsqueeze(0).expand(10, -1,-1)
        pred['seg_mask_ref'] = seg_mask_ref.unsqueeze(0).expand(10,-1)
        pred['input1'] = gt1[:, 0:3].unsqueeze(0).expand(10, -1, -1)
        pred['input2'] = gt2[:, 0:3].unsqueeze(0).expand(10, -1, -1)
        pred['seg_mask1'] = seg_mask[id1].unsqueeze(0).expand(10, -1)
        pred['seg_mask2'] = seg_mask[id2].unsqueeze(0).expand(10, -1)
        pred['anchors'] = new_params[..., :3]
        pred['shift'] = pcds['shift']
        pred['scale'] = pcds['scale']
        return pred

    def part_completion(self, x, seg_mask, id1, anchor_id, device):
        gt = x[id1]
        idx = gt[..., 3 + anchor_id] == 0.
        gt_part = gt[idx]
        latent = self.encoder(xyz=gt_part.unsqueeze(0))
        new_params, new_latents = self.decomposer(latent)
        latent_per_point = new_latents.reshape(1, self.num_anchors, 1, -1).expand(-1, -1, self.points_per_anchor, -1).reshape(1, self.npoints, -1)
        param_per_point = new_params.reshape(1, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(1, self.npoints, 3).transpose(1,2)
        assert param_per_point.shape[1] == 3
        anchor_per_point = param_per_point
        variance_per_point = None

        pred = self.decode(anchor_per_point, code=None, pointwise_latent=latent_per_point, device=device, variance=variance_per_point) 
        pred['input'] = gt_part[:, 0:3].unsqueeze(0)
        pred['ref'] = gt[:, 0:3].unsqueeze(0)
        pred['anchors'] = new_params[..., :3]
        
        return pred

    def combine_latent(self, x, seg_mask, ids, device):
        gts = []
        for i in ids:
            gts.append(x[i])
        gts = torch.stack(gts, dim=0)
        latents = self.encoder(xyz=gts)
        latents = torch.stack([latents[i, i] for i in range(self.num_anchors)], dim=0).reshape(1, self.num_anchors, -1)
        global_feature, new_params, new_latents = self.decomposer(latents)
        latent_per_point = new_latents.reshape(1, self.num_anchors, 1, -1).expand(-1, -1, self.points_per_anchor, -1).reshape(1, self.npoints, -1)
        param_per_point = new_params.reshape(1, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(1, self.npoints, 3).transpose(1,2)
        assert param_per_point.shape[1] == 3
        anchor_per_point = param_per_point
        variance_per_point = None
        
        pred = self.decode(anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, device=device, variance=variance_per_point) 
        ref = torch.cat([gts[i, seg_mask[ids[i]] == i, :3] for i in range(self.num_anchors)], dim=0)
        seg_mask_ref = torch.cat([seg_mask[i][seg_mask[ids[i]] == i] for i in range(self.num_anchors)], dim=0)
        pred['input_ref'] = ref.unsqueeze(0)
        pred['seg_mask_ref'] = seg_mask_ref.unsqueeze(0)
        for i in range(self.num_anchors):
            pred[f'ref_{i}'] = gts[i, :, 0:3].unsqueeze(0)
            pred[f'seg_mask_{i}'] = seg_mask[i].unsqueeze(0)
        pred['anchors'] = new_params[..., :3]
        return pred

    def forward(self, pcds, device='cuda'):
        """
            input: [B, npoints, 3]
            attn_map: [B, npoints, n_class]
            seg_mask: [B, npoints]
        """
        input = pcds['input'].to(device)
        ref = pcds['ref'].to(device)
        attn_map = pcds['attn_map'].to(device) # ->[B, npoints, n_class]
        seg_mask = pcds['seg_mask'].to(device)
        loss_dict = dict()

        npoint = input.shape[-2]
       

        if self.include_attn_weight_in_encoder:
            x = torch.cat([input, attn_map], dim=-1) # -> [B, npoints, n_class + 3]
        else:
            x = input

        B = input.shape[0]

        
        


        if self.training:
            part_latent = self.encoder(x)
            means, variances = self.parameter_regressor(part_latent)
            assert means.shape[1] == variances.shape[1] == self.num_anchors
            z = reparameterize_gaussian(mean=means, logvar=variances)  # (B, F)
            log_pz = standard_normal_logprob(z).sum(dim=2)  # (B, ), Independence assumption
            entropy = gaussian_entropy(logvar=variances)      # (B, )
            loss_prior = (- log_pz - entropy).mean()
            global_feature, anchors, part_latent = self.decomposer(z)
            loss_dict['prior_loss'] = self.kl_weight * loss_prior
            if self.guidance:
                idxs = torch.rand(part_latent.shape[:-1]).to(part_latent.device) >= self.part_latent_dropout_prob
                part_latent = part_latent * idxs.unsqueeze(-1)

                global_idxs = torch.rand(global_feature.shape[0]).to(part_latent.device) >= self.global_latent_dropout_prob
                global_feature = global_feature * global_idxs.unsqueeze(-1)

            t, _ = self.sampler.sample(B, device)
            anchor_assignments = seg_mask.to(torch.int32)

            latent_per_point = gather_operation(part_latent.transpose(2,1).contiguous(), anchor_assignments).reshape(B, -1, npoint)
            anchor_per_point = gather_operation(anchors.transpose(2, 1).contiguous(), anchor_assignments).reshape(B, 3, npoint)  
            assert anchor_per_point.shape[1] == 3

            anchor_loss = self.loss_func(anchor_per_point.transpose(1,2), ref)
            loss_dict['anchor_loss'] = self.anchor_loss_weight * anchor_loss

            diffusion_loss = self.diffusion.training_losses(ref.transpose(1,2), t, anchors=anchor_per_point, code=global_feature, pointwise_latent=latent_per_point.transpose(1,2), variance=None, reduce=True)
            loss_dict['diffusion_loss'] = diffusion_loss
            return loss_dict

        else:
            z = torch.randn([B, self.num_anchors, self.latent_dim]).cuda()
            
            
            if self.fix_part:
                anchor_id = 0
                z = torch.randn([1, self.num_anchors, self.latent_dim]).cuda().repeat_interleave(B, dim=0)
                z[:, anchor_id] = torch.randn([B, self.latent_dim]).cuda()
            if self.interpolate and not self.training:
                id1, id2 = 22,107
                anchor_id= 2
                return self.interpolate_latent(x, seg_mask, id1, id2, anchor_id, device, pcds)
            if self.combine and not self.training:
                ids = [23, 114, 19,89]
                return self.combine_latent(x, seg_mask, ids, device)
            global_feature, anchors, part_latent = self.decomposer(z)
            anchor_per_point = anchors.reshape(B, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, -1).transpose(1,2)
            assert anchor_per_point.shape[1] == 3
            latent_per_point = part_latent.reshape(B, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, -1)
            
            pred = self.q_sample(ref.transpose(2,1), anchors=anchor_per_point, device=device, variance=None) if self.forward_sample else self.decode(anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, device=device, variance=None) 
            pred['input'] = input
            pred['input_ref'] = ref
            pred['anchors'] = anchors
            pred['seg_mask'] = seg_mask 
            pred['seg_mask_ref'] = seg_mask 
            pred['assigned_anchor'] = anchor_per_point.transpose(1, 2)
            pred['shift'] = pcds['shift']
            pred['scale'] = pcds['scale']
            pred = {k:v.detach().cpu() for k, v in pred.items()}
            return pred
