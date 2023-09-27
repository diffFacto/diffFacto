import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Module, functional
from difffacto.models.diffusions.nets.attention import MLP
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS, DIFFUSIONS, SAMPLERS, METRICS, DECOMPOSERS, SEGMENTORS, GENERATORS, DISCRIMINATORS
from difffacto.metrics.common import gradient_penalty, gen_loss, dis_loss, triplet_loss
from pointnet2_ops.pointnet2_utils import gather_operation
from .language_utils.language_util import tokenizing
from einops import rearrange
VOCAB_SIZE=2787

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def identity(*args, **kwargs):
    return nn.Identity()

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@MODELS.register_module()
class AnchorDiffAE(Module):

    def __init__(
        self, 
        encoder, 
        diffusion, 
        sampler, 
        num_anchors,
        num_timesteps, 
        npoints=2048, 
        zero_anchors=False,
        gen=False,
        sample_noise_num=20,
        cimle=False,
        cimle_sample_num=10,
        diffusion_loss_weight=1.0,
        use_input=False,


        learn_var=False,
        detach_variance=True,
        detach_anchor=True,
        
        global_shift=False,
        global_scale=False,
        vertical_only=True,

        ret_traj=False, 
        ret_interval=20, 
        forward_sample=False,
        interpolate=False,
        interpolate_part_id=2,
        fix_part_ids = None,
        combine=False,
        drift_anchors=False,
        save_pred_xstart=False,
        save_dir=None,
        save_weights=False,
        
        
        noise_reg_loss=True,
        reg_loss_weight=1.0,
        
        pretrain_prior=False,
        train_language=False,
        language_encoder=None,
        clip_weight=1.0,
        triplet_weight=1.0,
        triplet_thresh=0.1,
        
        ):
        super().__init__()
        self.pretrain_prior=pretrain_prior
        self.encoder = build_from_cfg(encoder, ENCODERS)
        if pretrain_prior:
            self.diffusion = nn.ModuleList([build_from_cfg(diffusion, DIFFUSIONS, num_timesteps=num_timesteps) for _ in range(num_anchors)])
        else:
            self.diffusion = build_from_cfg(diffusion, DIFFUSIONS, num_timesteps=num_timesteps)
        self.sampler = build_from_cfg(sampler, SAMPLERS, num_timesteps=num_timesteps)
        self.diffusion_loss_weight=diffusion_loss_weight
        self.sample_noise_num=sample_noise_num
        self.interpolate_part_id=interpolate_part_id
        self.cimle=cimle
        self.cimle_sample_num=cimle_sample_num
        self.fix_part_ids=fix_part_ids
        self.learn_var = learn_var
        self.gen=gen
        self.use_input=use_input
        self.save_weights=save_weights
        self.save_dir=save_dir
        self.global_shift = global_shift
        self.global_scale=global_scale
        self.vertical_only=vertical_only
        self.zero_anchors=zero_anchors

        self.drift_anchors=drift_anchors
        self.num_timesteps = int(num_timesteps)
        self.num_anchors = num_anchors
        
        self.detach_anchor=detach_anchor
        self.detach_variance=detach_variance
        self.fixed_id= [0, 0, 0, 0]
        assert len(self.fixed_id) == self.num_anchors
        
        self.train_language=train_language
        if train_language:
            self.language_encoder = build_from_cfg(language_encoder, ENCODERS)
            self.update_mlp = zero_module(MLP([self.language_encoder.text_dim + self.encoder.zdim, self.encoder.zdim * 2, self.encoder.zdim*2, self.encoder.zdim], use_linear=True))
            self.temp = nn.Parameter(torch.zeros(1))
            self.clip_weight=clip_weight
            self.triplet_weight=triplet_weight
            self.triplet_thresh=triplet_thresh
        
        self.npoints = npoints
        self.points_per_anchor = self.npoints // self.num_anchors
        self.forward_sample = forward_sample
        self.interpolate=interpolate
        self.ret_traj=ret_traj
        self.ret_interval=ret_interval
        self.save_pred_xstart=save_pred_xstart
        self.combine=combine
        self.noise_reg_loss=noise_reg_loss
        self.reg_loss_weight=reg_loss_weight
        
        # lgan training
        print(self)

    def train_language_module(self):
        self.language_encoder.train()
        self.update_mlp.train()
        
    def eval_language_module(self):
        self.language_encoder.eval()
        self.update_mlp.eval()
        
    def decode(self, anchors, ctx=None, noise=None, variance=None, anchor_assignments=None, valid_id=None, device='cuda'):
        final = dict() 
        np = anchor_assignments.shape[1]
        bs = anchors.shape[0]
        for t, sample in self.diffusion.p_sample_loop_progressive(
            [bs, 3, np],
            anchors=anchors,
            variance=variance,
            ctx=ctx,
            noise=noise,
            anchor_assignment=anchor_assignments,
            valid_id=valid_id,
            device=device,
            progress=False,
        ):
            if t == 0:
                final['pred'] = sample['sample'].transpose(2, 1)
                if self.save_pred_xstart:
                    final['pred_xstart'] = sample['pred_xstart'].transpose(1,2) 
            elif (self.ret_traj and t % self.ret_interval == 0):
                final[t] = sample['sample'].transpose(2, 1) 
                if self.save_pred_xstart:
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
    @torch.no_grad()
    def interpolate_two_shapes(self, pcds1, pcds2, part_id, param_shift, param_scale, mid_num=10):
        B = pcds1.shape[0]
        xyz1, mask1 = torch.split(pcds1, [3,1], dim=-1)
        xyz2, mask2 = torch.split(pcds2, [3,1], dim=-1)
        # import pickle
        # pickle.dump(dict(pred=xyz1.cpu().numpy(), pred_seg_mask=mask1.cpu().numpy()), open("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/out.pkl", "wb"))
        # exit()
        mask1 = mask1.reshape(B, -1).long()
        mask2 = mask2.reshape(B, -1).long()
        valid_id = torch.zeros(B, 4).cuda()
        for i in range(4):
            m = torch.any(mask1 == i, dim=1)
            valid_id[m] = 1
        anchor_per_point, ctx, variance_per_point, anchor_assignments, valid_id = self.encoder.interpolate_two_shape(xyz1, mask1, xyz2, mask2, self.npoints, part_id, param_shift, param_scale, valid_id, mid_num)
        pred = self.decode(anchor_per_point, ctx=ctx, device='cuda', variance=variance_per_point, anchor_assignments=anchor_assignments, valid_id=valid_id) 
        return pred['pred'].reshape(B, mid_num, self.npoints, 3), anchor_assignments.reshape(B, mid_num, self.npoints)
        
    def interpolate_latent(self, device, pcds):
        out_dict = {}
        ref = pcds['ref'].to(device)
        input = pcds['input'].to(device)
        ref_seg_mask = pcds['ref_seg_mask'].to(device)
        seg_flag = pcds['attn_map'].to(device)
        valid_id = pcds['present'].to(device)
        gt_part_shift = pcds.get('part_shift', torch.zeros_like(ref).transpose(1,2))
        gt_part_var = pcds.get('part_scale', torch.ones_like(ref).transpose(1,2))
        B = input.shape[0]
        if not self.encoder.origin_scale:
            gt_part_var = gt_part_var ** 2
        if self.cimle:
            noise, _ = self.encoder.sample_noise(pcds, device, 1)
            noise = noise.squeeze(1)
        else:
            noise = None
        if self.gen:
            part_code = torch.randn(B, self.encoder.zdim, self.num_anchors).to(device) * math.sqrt(self.encoder.prior_var)
            if self.encoder.use_flow:
                part_code_ = []
                for i in range(self.num_anchors):
                    _part_code = part_code[..., i]
                    part_code_.append(self.encoder.flow[i](_part_code, reverse=True).view(B, self.encoder.zdim))
                part_code = torch.stack(part_code_, dim=-1)
            valid_id[..., self.interpolate_part_id] = 1.
            pred_seg_mask = torch.arange(self.num_anchors).cuda()[None] * valid_id + torch.argmax(valid_id, dim=1, keepdims=True) * (1 - valid_id)
            pred_seg_mask = pred_seg_mask.repeat_interleave(self.npoints // self.num_anchors, dim=1)
        else:
            part_code_means, part_code_logvars = self.encoder.get_part_code(input, seg_flag)
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)
            pred_seg_mask = ref_seg_mask.reshape(-1, ref_seg_mask.shape[1], 1).expand(-1, -1, self.npoints // ref_seg_mask.shape[1]).reshape(-1, self.npoints)
            
        K = 10
        dx = torch.linspace(0, 1, steps=K).reshape(1, -1, 1).to(device)
        part_code_perm = torch.randperm(B)
        permuted_ref = ref[part_code_perm]
        permuted_ref_seg_mask = ref_seg_mask[part_code_perm]
        interpolated_part_code = part_code[..., self.interpolate_part_id].unsqueeze(1) + (part_code[part_code_perm,:, self.interpolate_part_id].unsqueeze(1) - part_code[..., self.interpolate_part_id].unsqueeze(1)) * dx
        part_code = part_code.unsqueeze(1).repeat_interleave(K, dim=1)
        part_code[..., self.interpolate_part_id] = interpolated_part_code
        part_code = part_code.reshape(B*K, -1, self.num_anchors)
        gt_part_shift = gt_part_shift.repeat_interleave(K, dim=0)
        gt_part_var = gt_part_var.repeat_interleave(K, dim=0)
        noise = noise.repeat_interleave(K, dim=0)
        valid_id = valid_id.repeat_interleave(K, dim=0)
        mean, logvar = self.encoder.get_params_from_part_code(part_code, valid_id, gt_mean=gt_part_shift, gt_var=gt_part_var, noise=noise)
            
        anchor_assignments = pred_seg_mask.repeat_interleave(K, dim=0).int()
        anchor_per_point, logvar_per_point, _ = self.encoder.gather_all(anchor_assignments, anchors=mean, variances=logvar)
        ctx = self.encoder.prepare_ctx(part_code, mean, logvar, anchor_assignments=anchor_assignments)
        
        N = anchor_assignments.shape[1]
        if self.npoints < N:
            anchor_per_point, logvar_per_point  = map(lambda r: r[..., :self.npoints], [anchor_per_point, logvar_per_point])
            ctx = [r[..., self.npoints] if r.shape[-1] > self.npoints else r for r in ctx]
            ref = ref[:, :self.npoints]
            anchor_assignments = anchor_assignments[:, :self.npoints]
        variance_per_point = torch.exp(logvar_per_point)
        priors = torch.randn_like(variance_per_point).transpose(1,2).to(variance_per_point.device) * variance_per_point.transpose(1,2) + anchor_per_point.transpose(1,2)
        from tqdm import tqdm
        from collections import defaultdict
        pred = defaultdict(list)
        chunk = 50
        # B = 5
        for k in tqdm(range(0, B*K, chunk)):
            anchor_per_point_chunk = anchor_per_point[k: k+chunk]
            ctx_chunk = [c[k: k+chunk] for c in ctx]
            variance_per_point_chunk = variance_per_point[k: k+chunk]
            anchor_assignments_chunk = anchor_assignments[k: k+chunk]
            valid_id_chunk = valid_id[k: k+chunk]
            # import ipdb; ipdb.set_trace()
            pred_chunk = self.decode(anchor_per_point_chunk, ctx=ctx_chunk, device=device, variance=variance_per_point_chunk, anchor_assignments=anchor_assignments_chunk, valid_id=valid_id_chunk) 
            pred_chunk = {key: v.detach().cpu() for key, v in pred_chunk.items()}
            for key in pred_chunk.keys():
                pred[key].append(pred_chunk[key].detach().cpu())
        # ref = torch.zeros([self.npoints, 3]).cuda()
        # _ref = torch.cat([gt1[seg_mask[id1] != anchor_id, :3], gt2[seg_mask[id2] == anchor_id, :3]], dim=0)
        # ref[:_ref.shape[0]] = _ref
        # seg_mask_ref = torch.zeros_like(seg_mask[0])
        # _seg_mask_ref = torch.cat([seg_mask[id1][seg_mask[id1] != anchor_id], seg_mask[id2][seg_mask[id2] == anchor_id]], dim=0)
        # seg_mask_ref[:_seg_mask_ref.shape[0]] = _seg_mask_ref
        pred = {key: torch.cat(v) for key, v in pred.items()}
        _pred = pred['pred'].reshape(B, K, self.npoints, 3)
        # priors = priors.reshape(B, K, self.npoints, 3)
        
        for i in range(K):
            out_dict[f"interpolate sample {i}"] = _pred[:,i]
        # for i in range(K):
        #     out_dict[f"interpolate sample prior {i}"] = priors[:, i]
            
        out_dict['pred_seg_mask'] = pred_seg_mask[:B]
        out_dict['ref_seg_mask'] = ref_seg_mask
        out_dict['pred'] = out_dict["interpolate sample 0"]
        out_dict['input_ref'] = ref
        out_dict['permuted_ref'] = permuted_ref
        out_dict['permuted_ref_seg_mask'] = permuted_ref_seg_mask
        out_dict['shift'] = pcds['shift']
        out_dict['scale'] = pcds['scale']
        return out_dict
    
    def sample_one_part(self, code, valid_id, mean, logvar, seg_mask, part_id, sample_num_each, fix_size, param_sample_num, selective_param_sample): 
        bs = code.shape[0]
        np = seg_mask.shape[1]
        ctx, \
        mean_per_point, \
        logvar_per_point, \
        seg_mask, \
        valid_id, \
        latents = \
            self.encoder.sample_with_fixed_latents(
            code,
            valid_id,
            mean, 
            logvar,
            seg_mask, 
            part_id, 
            sample_num_each,
            fix_size, 
            param_sample_num,
            selective_param_sample
        )
        codes, noise_latents, means, logvars = latents
        variance_per_point = torch.exp(logvar_per_point)
        pred = self.decode(mean_per_point, ctx=ctx, device='cuda', variance=variance_per_point, anchor_assignments=seg_mask.to(torch.int32), valid_id=valid_id) 
        pred = pred['pred'].reshape(bs, sample_num_each, param_sample_num, np, 3)
        seg_mask = seg_mask.reshape(bs, sample_num_each, param_sample_num, np)
        valid_id = valid_id.reshape(bs, sample_num_each, param_sample_num, self.num_anchors)
        codes = codes.reshape(bs, sample_num_each, param_sample_num, self.encoder.zdim, self.num_anchors)
        means = means.reshape(bs, sample_num_each, param_sample_num, 3, self.num_anchors)
        logvars = logvars.reshape(bs, sample_num_each, param_sample_num, 3, self.num_anchors)
        return pred, seg_mask, valid_id, codes, noise_latents, means, logvars
    def interpolate_params(self, device, pcds):
        out_dict = {}
        ref = pcds['ref'].to(device)
        input = pcds['input'].to(device)
        ref_seg_mask = pcds['ref_seg_mask'].to(device)
        seg_flag = pcds['attn_map'].to(device)
        valid_id = pcds['present'].to(device)
        gt_part_shift = pcds.get('part_shift', torch.zeros_like(ref).transpose(1,2))
        gt_part_var = pcds.get('part_scale', torch.ones_like(ref).transpose(1,2))
        B = input.shape[0]
        if not self.encoder.origin_scale:
            gt_part_var = gt_part_var ** 2
        if self.cimle:
            noise, _ = self.encoder.sample_noise(pcds, device, 1)
            noise = noise.squeeze(1)
        else:
            noise = None
        part_code_means, part_code_logvars = self.encoder.get_part_code(input, seg_flag)
        if self.encoder.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)  # (B, F)
        else:
            part_code = part_code_means.transpose(1,2)
            
        K = self.cimle_sample_num
        dx = torch.linspace(1, 5, steps=K).reshape(1, -1).to(device).expand(B, -1).reshape(B*K, 1)
        part_code = part_code.repeat_interleave(K, dim=0)
        gt_part_shift = gt_part_shift.repeat_interleave(K, dim=0)
        gt_part_var = gt_part_var.repeat_interleave(K, dim=0)
        noise = noise.repeat_interleave(K, dim=0)
        valid_id = valid_id.repeat_interleave(K, dim=0)
        mean, logvar = self.encoder.get_params_from_part_code(part_code, valid_id, gt_mean=gt_part_shift, gt_var=gt_part_var, noise=noise)
        mean[:, 1, [0, 2]] = (mean[:, 1, [0, 2]] * torch.sqrt(dx))
        logvar[:, 1, [0, 2]] = (logvar[:, 1, [0, 2]] + torch.log(dx))
            
        
        
        anchor_assignments = ref_seg_mask.repeat_interleave(K, dim=0).int()
        anchor_per_point, logvar_per_point, _ = self.encoder.gather_all(anchor_assignments, anchors=mean, variances=logvar + self.encoder.log_scale_var if logvar is not None else logvar)
        ctx = self.encoder.prepare_ctx(part_code, mean, logvar, anchor_assignments=anchor_assignments)
        
        N = anchor_assignments.shape[1]
        if self.npoints < N:
            anchor_per_point, logvar_per_point  = map(lambda r: r[..., :self.npoints], [anchor_per_point, logvar_per_point])
            ctx = [r[..., self.npoints] if r.shape[-1] > self.npoints else r for r in ctx]
            ref = ref[:, :self.npoints]
            anchor_assignments = anchor_assignments[:, :self.npoints]
        variance_per_point = torch.exp(logvar_per_point)
        priors = torch.randn(B*K, self.npoints, 3).to(variance_per_point.device) * variance_per_point.transpose(1,2) + anchor_per_point.transpose(1,2)
        

        pred = self.decode(anchor_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=anchor_assignments, valid_id=valid_id) 
        # ref = torch.zeros([self.npoints, 3]).cuda()
        # _ref = torch.cat([gt1[seg_mask[id1] != anchor_id, :3], gt2[seg_mask[id2] == anchor_id, :3]], dim=0)
        # ref[:_ref.shape[0]] = _ref
        # seg_mask_ref = torch.zeros_like(seg_mask[0])
        # _seg_mask_ref = torch.cat([seg_mask[id1][seg_mask[id1] != anchor_id], seg_mask[id2][seg_mask[id2] == anchor_id]], dim=0)
        # seg_mask_ref[:_seg_mask_ref.shape[0]] = _seg_mask_ref
        _pred = pred['pred'].reshape(B, K, self.npoints, 3)
        priors = priors.reshape(B, K, self.npoints, 3)
        
        for i in range(K):
            out_dict[f"interpolate sample {i}"] = _pred[:,i]
        for i in range(K):
            out_dict[f"interpolate sample prior {i}"] = priors[:, i]
            
        out_dict['pred_seg_mask'] = ref_seg_mask
        out_dict['ref_seg_mask'] = ref_seg_mask
        out_dict['seg_mask'] = ref_seg_mask
        out_dict['pred'] = out_dict["interpolate sample 0"]
        out_dict['input_ref'] = ref
        out_dict['shift'] = pcds['shift']
        out_dict['scale'] = pcds['scale']
        return out_dict

    def combine_latent_specific(self, inputs, device):
        # idss = [[61, 51, 49, 40, 38, 25, 23, 21, 13], [51, 34, 12], [62, 56, 54, 38, 29, 28, 13, 11, 7], [42, 36, 34, 32, 62]]
        # idss = [[21, 34, 13, 62], [61, 51, 62, 42], [61, 34, 56, 42], [61, 51, 56, 42], [51, 12, 38, 36], [51, 34, 29, 42], [51, 34, 11, 62]]
        assert len(inputs) == self.num_anchors
        valid_id = torch.tensor([ 1 if torch.any(inp !=0) else 0 for i, inp in enumerate(inputs) ]).cuda().unsqueeze(0)
        seg_flag = torch.cat([torch.eye(self.num_anchors).cuda()[i].reshape(1, self.num_anchors).repeat_interleave(inp.shape[0], dim=0) for i, inp in enumerate(inputs) if torch.any(inp !=0)], dim=0).unsqueeze(0)
        inputs = [inp for inp in inputs if torch.any(inp !=0)]
        input = torch.cat(inputs, dim=0).unsqueeze(0)
        print(seg_flag.shape, input.shape, valid_id)
        part_code_means, part_code_logvars = self.encoder.get_part_code(input, seg_flag)
        part_code = part_code_means.transpose(1,2)
        
        if self.cimle:
            K=self.cimle_sample_num
            noise = torch.randn(K, self.encoder.part_aligner.noise_dim).cuda()
            noise = noise.reshape(K, -1)
        else:
            K=1
            noise = None
        part_code, valid_id = map(lambda a: a.repeat_interleave(K, dim=0), [part_code, valid_id])
        out = self.encoder.get_params_from_part_code(part_code, valid_id, gt_mean=None, gt_var=None, noise=noise)
        if len(out) == 3:
            mean, logvar, _ = out
        else:
            mean, logvar = out
        anchor_assignments = torch.arange(self.num_anchors).unsqueeze(0).to(device) * valid_id 
        anchor_assignments = anchor_assignments.repeat_interleave(self.npoints // self.num_anchors, dim=1).to(torch.int32)
        mean_per_point, logvar_per_point, _ = self.encoder.gather_all(anchor_assignments, anchors=mean, variances=logvar + self.encoder.log_scale_var if logvar is not None else logvar)
        variance_per_point = torch.exp(logvar_per_point)
        ctx = self.encoder.prepare_ctx(part_code, mean, logvar, anchor_assignments=anchor_assignments)
        _pred = self.decode(mean_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=anchor_assignments, valid_id=valid_id) 
        pred = dict()
        for k, v in _pred.items():
            v = rearrange(v, "(b k) ... -> b k ...", k=K)
            for i in range(K):
                pred[f"{k}_sample_{i}"] = v[:, i]
        pred['pred'] = pred['pred_sample_0']
        pred['input'] = input
        pred['pred_seg_mask'] = anchor_assignments
        pred['seg_mask'] = torch.argmax(seg_flag, dim=2)
            
        pred['shift'] = torch.zeros(1, 1, 3).cuda()
        pred['scale'] = torch.ones(1, 1, 1).cuda()
        return pred

    def combine_latent(self, pcds, device):
        # idss = [[61, 51, 49, 40, 38, 25, 23, 21, 13], [51, 34, 12], [62, 56, 54, 38, 29, 28, 13, 11, 7], [42, 36, 34, 32, 62]]
        # idss = [[21, 34, 13, 62], [61, 51, 62, 42], [61, 34, 56, 42], [61, 51, 56, 42], [51, 12, 38, 36], [51, 34, 29, 42], [51, 34, 11, 62]]
        ref = pcds['ref'].to(device)
        refs = [ref for _ in range(self.num_anchors)]
        input = pcds['input'].to(device)
        seg_flag = pcds['attn_map'].to(device)
        refs_seg_mask = [pcds['ref_seg_mask'] for _ in range(self.num_anchors)]
        valid_id = pcds['present'].to(device)  
        gt_part_shift = pcds.get('part_shift', torch.zeros_like(ref).transpose(1,2))
        gt_part_var = pcds.get('part_scale', torch.ones_like(ref).transpose(1,2))
        B = input.shape[0]
        if not self.encoder.origin_scale:
            gt_part_var = gt_part_var ** 2
        
        
        part_code_means, part_code_logvars = self.encoder.get_part_code(input, seg_flag)
        if self.encoder.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)  # (B, F)
        else:
            part_code = part_code_means.transpose(1,2)
        
        if self.cimle:
            if self.encoder.selective_noise_sampling:
                K = 100
            else:
                K=self.cimle_sample_num
            noise, _ = self.encoder.sample_noise(pcds, device, K)
            noise = noise.reshape(B*K, -1)
        else:
            K=1
            noise = None
        for i in range(self.num_anchors):
            perm = torch.randperm(B)
            refs[i] = refs[i][perm]
            refs_seg_mask[i] = pcds['ref_seg_mask'][perm]
            part_code[..., i] = part_code[perm][..., i]
            valid_id[:, i] = valid_id[perm][..., i] * valid_id[:, i]
            gt_part_shift[..., i] = gt_part_shift[perm][..., i]
            gt_part_var[..., i] = gt_part_var[perm][..., i]
        
        part_code, valid_id, gt_part_shift, gt_part_var = map(lambda a: a.repeat_interleave(K, dim=0), [part_code, valid_id, gt_part_shift, gt_part_var])
        out = self.encoder.get_params_from_part_code(part_code, valid_id, gt_mean=gt_part_shift, gt_var=gt_part_var, noise=noise)
        if len(out) == 3:
            mean, logvar, _ = out
        else:
            mean, logvar = out
        
        if self.encoder.selective_noise_sampling:
            mean, logvar = self.encoder.subsample_params(mean.reshape(B, K, 3, self.num_anchors), logvar.reshape(B, K, 3, self.num_anchors), valid_id.reshape(B, K, self.num_anchors)[:, 0], num=10)
            K = 10
            valid_id = valid_id.reshape(B, 100, self.num_anchors)[:, :K].reshape(B*K, self.num_anchors)
            part_code = part_code.reshape(B, 100, -1, self.num_anchors)[:, :K].reshape(B*K, -1, self.num_anchors)
        anchor_assignments = torch.arange(self.num_anchors).unsqueeze(0).to(device) * valid_id  + torch.argmax(valid_id, dim=1).unsqueeze(1) * (1 - valid_id)
        anchor_assignments = anchor_assignments.repeat_interleave(self.npoints // self.num_anchors, dim=1).to(torch.int32)
        mean_per_point, logvar_per_point, _ = self.encoder.gather_all(anchor_assignments, anchors=mean, variances=logvar + self.encoder.log_scale_var if logvar is not None else logvar)
        variance_per_point = torch.exp(logvar_per_point)
        ctx = self.encoder.prepare_ctx(part_code, mean, logvar, anchor_assignments=anchor_assignments)

        _pred = self.decode(mean_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=anchor_assignments, valid_id=valid_id) 
        pred = dict()
        for k, v in _pred.items():
            v = rearrange(v, "(b k) ... -> b k ...", k=K)
            for i in range(K):
                pred[f"{k}_sample_{i}"] = v[:, i]
        pred['pred'] = pred['pred_sample_0']
        pred['pred_seg_mask'] = anchor_assignments.reshape(B, K, -1)[:, 0]
        pred['input_ref'] = ref
        pred['ref_seg_mask'] = pcds['ref_seg_mask']
        for i in range(self.num_anchors):
            pred[f'input_ref{i}'] = refs[i]
            pred[f"ref_seg_mask{i}"] = refs_seg_mask[i]
            
        pred['shift'] = pcds['shift']
        pred['scale'] = pcds['scale']
        return pred

    def language_train_step(self, data, device):
        input_pcds = data['input'].to(device)
        ref_pcds = data['ref'].to(device)
        attn_maps = data['attn_map'].to(device)
        ref_attn_maps = data['ref_attn_map'].to(device)
        seg_mask = data['seg_mask'].to(device)
        ref_seg_mask = data['ref_seg_mask'].to(device)
        part_indicators = data['part_indicator'].to(device)
        texts = data['text'].to(device)
        B,_, npoint = input_pcds.shape[:-1]
        language_enc_f, language_enc_attn = self.language_encoder(texts)
    
        part_code_means, part_code_logvars = self.encoder.get_part_code(input_pcds.reshape(B*3, npoint, 3), attn_maps.reshape(B*3, npoint, self.num_anchors))
        if self.encoder.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2).reshape(B, 3, self.encoder.zdim, self.num_anchors)  # (B, F)
        else:
            part_code = part_code_means.transpose(1,2).reshape(B, 3, self.encoder.zdim, self.num_anchors)
            
        

        part_id = part_indicators.max(-1)[1].to(torch.int)
        referenced_part_latent = gather_operation(part_code.reshape(B*3, self.encoder.zdim, self.num_anchors).contiguous(), part_id.reshape(B*3, 1)).reshape(B, 3, self.encoder.zdim)
        tgt_latent, latent_for_edit, distractor_latent = referenced_part_latent.split(1, dim=1)
        tgt_latent, latent_for_edit, distractor_latent = tgt_latent.reshape(B, self.encoder.zdim), latent_for_edit.reshape(B, self.encoder.zdim), distractor_latent.reshape(B, self.encoder.zdim)
        modified_part_latent_res = self.update_mlp(torch.cat([language_enc_f, latent_for_edit], -1))
        modified_part_latent = latent_for_edit + modified_part_latent_res
        loss_dict = dict()
        # clip loss
        cos_sim = F.normalize(modified_part_latent, dim=1)[..., None] * F.normalize(torch.stack([tgt_latent, distractor_latent], dim=-1), dim=1)# * torch.exp(self.temp).reshape(1, 1, 1)
        cos_sim = cos_sim.sum(1)
        label = torch.tensor([1,0]).cuda().reshape(1, 2).expand(B, -1).float()
        clip_l = F.binary_cross_entropy(F.sigmoid(cos_sim), label)
        # triplet loss
        loss_dict['clip_loss'] = self.clip_weight * clip_l
        loss_dict['cos_sim_pos'] = cos_sim[:, 0]
        loss_dict['cos_sim_neg'] = cos_sim[:, 1]
        trip_l, pos_diff, neg_diff = triplet_loss(modified_part_latent, tgt_latent, distractor_latent, thresh=self.triplet_thresh) 
        loss_dict['triplet_loss'] = self.triplet_weight * trip_l
        loss_dict['pos_diff'] = pos_diff
        loss_dict['neg_diff'] = neg_diff
        return loss_dict
    
    @torch.no_grad()
    def language_edit_step(self, data, device):
        prompts = [
            "leg",
            "four leg", 
            "one leg",
            "thick leg", 
            "thin leg", 
            "wheel", 
            "long leg", 
            "short leg", 
            "seat", 
            "thin seat", 
            "thick seat",
            "back",
            "straight back", 
            "slanted back",
            "tall back", 
            "short back",
            "square back",
            "diamond back", 
            "arm rest", 
        ]
        # ids = [2]
        ids = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0,0, 0, 3]
        prompts = [f"a chair with {p}" for p in prompts]
        tokenized_prompts = [tokenizing(f"a chair with {p}") for p in prompts]
        # if self.partglot_dataset:
        #     target_pcds = data['target'].to(device)
        #     target_attn_maps = data['target_attn_map'].to(device)
        #     distractor_pcds = data['distractor'].to(device)
        #     distractor_attn_maps = data['distractor_attn_map'].to(device)
        #     part_indicators = data['part_indicator'].to(device).repeat_interleave(3, dim=0)
        #     texts = data['text'].to(device)
        #     B, npoint = target_pcds.shape[:-1]
        #     pcds = torch.stack([target_pcds, distractor_pcds, distractor_pcds], dim=1).reshape(B*3, npoint, 3)
        #     attn_maps = torch.stack([target_attn_maps, distractor_attn_maps, distractor_attn_maps], dim=1).reshape(B*3, npoint, -1)
        #     input = torch.cat([pcds, attn_maps], dim=-1)
        #     language_enc_f, language_enc_attn = self.language_encoder(texts)
        
        
        #     part_latent = self.encoder(input)
        #     n_part, n_dim = part_latent.shape[1:]
            

        #     part_id = part_indicators.max(1)[1].to(torch.int)
        #     referenced_part_latent = gather_operation(part_latent.transpose(1,2).contiguous(), part_id.reshape(B*3, 1)).reshape(B, 3, 1, -1)
        #     tgt_latent, latent_for_edit, distractor_latent = referenced_part_latent.reshape(B, 3, 1, n_dim).split(1, dim=1)
        #     tgt_latent, latent_for_edit, distractor_latent = tgt_latent.reshape(B, n_dim), latent_for_edit.reshape(B, n_dim), distractor_latent.reshape(B, n_dim)
        #     modified_part_latent_res = self.update_mlp(torch.cat([language_enc_f, latent_for_edit], -1))
        #     modified_part_latent = latent_for_edit + modified_part_latent_res
        #     modified_part_latent = torch.stack([tgt_latent, modified_part_latent, distractor_latent], dim=1).reshape(B*3, 1, n_dim)
        #     total_modified_part_latent = (1 - part_indicators).reshape(B*3, n_part, 1) * part_latent + part_indicators.reshape(B*3, n_part, 1) * modified_part_latent
        #     global_feature, anchors, part_latent = self.decomposer(total_modified_part_latent)
            
            
        #     assert anchors.shape[1] == part_latent.shape[1] == self.num_anchors
        #     param_per_point = anchors.reshape(B*3, self.num_anchors, 1, 3).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*3, self.npoints, 3).transpose(1,2)
        #     assert param_per_point.shape[1] == 3
        #     anchor_per_point = param_per_point

        #     latent_per_point = part_latent.reshape(B*3, self.num_anchors, 1, n_dim).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*3, self.npoints, n_dim)
        #     pred = self.decode(anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, device=device, variance=None) 
        #     pred_out = dict()
        #     for k, v in pred.items():
        #         target, edit, distractor = v.reshape(B, 3, self.npoints, 3).split(1, dim=1)
        #         pred_out[f"distractor_{k}"] = distractor.squeeze(1)
        #         pred_out[f"distractor_edit_{k}"] = edit.squeeze(1)
        #         pred_out[f"target_{k}"] = target.squeeze(1)
            

        #     pred_out['text'] = texts
        #     pred_out['input_distractor'] = distractor_pcds
        #     pred_out['input_target'] = target_pcds
        #     pred_out['anchors'] = anchors
        #     pred_out['seg_mask'] = distractor_attn_maps.max(-1)[1]
        #     pred_out['seg_mask_ref'] = target_attn_maps.max(-1)[1]
        #     pred_out['assigned_anchor'] = anchor_per_point.transpose(1, 2)
        #     pred_out['target_shift'] = data['target_shift']
        #     pred_out['target_scale'] = data['target_scale']
        #     pred_out['distractor_shift'] = data['distractor_shift']
        #     pred_out['distractor_scale'] = data['distractor_scale']
        #     return pred_out
        # else:
        input = data['input'].to(device)
        ref = data['ref'].to(device)
        attn_map = data['attn_map'].to(device) # ->[B, npoints, n_class]
        seg_mask = data['seg_mask'].to(device)
        valid_id = data['present'].to(device).float()
        shift = data['shift'].to(device)
        scale = data['scale'].to(device)
        out_dict = dict()
        B = input.shape[0]
        if len(input.shape) == 4:
            input = input[:, 1]
            ref = ref[:, 1]
            attn_map = attn_map[:, 1]
            valid_id=valid_id[:,1]
            seg_mask=seg_mask[:, 1]
            shift = shift[:, 1]
            scale = scale[:, 1]
        part_code_means, part_code_logvars = self.encoder.get_part_code(input, attn_map)
        if self.encoder.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2).reshape(B, self.encoder.zdim, self.num_anchors)  # (B, F)
        else:
            part_code = part_code_means.transpose(1,2).reshape(B, self.encoder.zdim, self.num_anchors)
        part_latent_for_edit = part_code
        for i, prompt in enumerate(tokenized_prompts):
            part_latent_edit = part_latent_for_edit.clone()
            print(prompts[i])
            prompt = prompt.to(device).unsqueeze(0)
            edit_id = ids[i]
            language_enc_f, language_enc_attn = self.language_encoder(prompt)
            language_enc_f = language_enc_f.repeat_interleave(B, dim=0)
            referenced_part_latent = part_latent_for_edit[..., edit_id]
            if self.cimle:
                K=self.cimle_sample_num
                noise = torch.randn([B*K, self.encoder.part_aligner.noise_dim]).cuda()
            else:
                noise = None
            
            modified_part_latent = self.update_mlp(torch.cat([language_enc_f, referenced_part_latent], -1)) + referenced_part_latent
            part_latent_edit[..., edit_id] = modified_part_latent
            part_latent = torch.cat([part_latent_for_edit, part_latent_edit], dim=0)
            _valid_id = valid_id[None].repeat_interleave(2, dim=0).reshape(2, B, self.num_anchors)
            noise = noise[None].repeat_interleave(2, dim=0).reshape(2*B*K, self.encoder.part_aligner.noise_dim)
            _valid_id[1, :, edit_id] = 1
            _valid_id = _valid_id.reshape(2*B, self.num_anchors).repeat_interleave(K, dim=0)
            part_latent = part_latent.repeat_interleave(K, dim=0)
            mean, logvar = self.encoder.get_params_from_part_code(part_latent, _valid_id, noise=noise)
            anchor_assignments = torch.arange(self.num_anchors).unsqueeze(0).cuda() * _valid_id + torch.argmax(_valid_id, dim=1, keepdim=True) * (1 - _valid_id)
            anchor_assignments = anchor_assignments.reshape(2*B, 1, self.num_anchors, 1).expand(-1, K, -1, self.npoints // self.num_anchors).reshape(2*B*K, self.npoints).int()
            anchor_per_point, logvar_per_point, _ = self.encoder.gather_all(anchor_assignments, anchors=mean, variances=logvar)
            ctx = self.encoder.prepare_ctx(part_latent, mean, logvar, anchor_assignments=anchor_assignments)
            variance_per_point = torch.exp(logvar_per_point)
            pred = self.decode(anchor_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=anchor_assignments, valid_id=_valid_id) 
            pred_pcd = pred['pred'].reshape(2, B, K, self.npoints, 3)
            for i in range(K):
                out_dict[f"{prompts[i]}_edit_sample_{i}"] = pred_pcd[1, :, i]
                out_dict[f"{prompts[i]}_original_sample_{i}"] = pred_pcd[0, :, i]
        out_dict['pred_seg_mask'] = anchor_assignments
        out_dict['ref_seg_mask'] = seg_mask
        out_dict['seg_mask'] = seg_mask
        out_dict['pred'] = ref
        out_dict['input_ref'] = ref
        out_dict['shift'] = shift
        out_dict['scale'] = scale
        return [(out_dict, "language")]

        # if self.sample_by_seg_mask:
        #     anchor_assignments = seg_mask.to(torch.int32)
        #     scale = int(self.npoints / anchor_assignments.shape[1])
        #     anchor_assignments = anchor_assignments.repeat_interleave(scale, dim=1)
        #     input_latent_per_point = gather_operation(input_part_latent.transpose(2,1).contiguous(), anchor_assignments).reshape(B, -1, self.npoints).transpose(1,2)
        #     input_anchor_per_point = gather_operation(input_anchor.transpose(2, 1).contiguous(), anchor_assignments).reshape(B, 3, self.npoints)
        #     assert input_anchor_per_point.shape[1] == 3
        # else:
        #     input_anchor_per_point = input_anchor.reshape(B, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, -1).transpose(1,2)
        #     assert input_anchor_per_point.shape[1] == 3
        #     input_latent_per_point = input_part_latent.reshape(B, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, -1)
        
        # pred = self.decode(input_anchor_per_point, code=input_global_feature, pointwise_latent=input_latent_per_point, device=device, variance=None) 
        # pred['input'] = input
        # pred['input_ref'] = ref
        # pred['anchors'] = input_anchor
        # pred['seg_mask'] = seg_mask 
        # pred['seg_mask_ref'] = seg_mask 
        # pred['assigned_anchor'] = input_anchor_per_point.transpose(1, 2)
        # pred['shift'] = data['shift']
        # pred['scale'] = data['scale']

        # if self.latent_diffusion is not None:
        #     part_latent_for_edit = part_latent_for_edit.transpose(1,2)
        #     input_anchor = input_anchor.transpose(1,2)
        # for i, prompt in enumerate(tokenized_prompts):
        #     print(prompts[i])
        #     prompt = prompt.to(device).unsqueeze(0)
        #     one_hot_id = torch.eye(self.num_anchors).to(device)[ids[i]].unsqueeze(0).repeat_interleave(B, dim=0)
        #     language_enc_f, language_enc_attn = self.language_encoder(prompt)
        #     language_enc_f = language_enc_f.repeat_interleave(B, dim=0)
        #     if self.latent_diffusion is not None:
                
        #         language_enc_f = language_enc_f.transpose(1,2)
        #         # print(part_latent_for_edit.shape, language_enc_f.shape)
        #         total_modified_part_latent = self.latent_diffusion.p_sample_loop(part_latent_for_edit.shape, part_latent_for_edit, language_enc_f, anchors=input_anchor)
        #         # print(total_modified_part_latent.shape)
        #         total_modified_part_latent = total_modified_part_latent.transpose(1, 2).unsqueeze(1)
        #         K = total_modified_part_latent.shape[1]
        #         # total_modified_part_latent = part_latent_for_edit + torch.rand_like(part_latent_for_edit)
        #     else:
        #         referenced_part_latent = part_latent_for_edit[:, ids[i]]
        #         if self.icmle:
        #             K=10
        #             conditional = torch.randn([B, K, self.conditional_dim]).cuda()
        #         else:
        #             K=1
        #             conditional = None
        #         total_modified_part_latent = self.latent_language_encoder(part_latent_for_edit, one_hot_id, language_enc_f, conditional=conditional)
        #     edited_global_feature, edited_anchors, edited_part_latent = self.decomposer(total_modified_part_latent.reshape(B*K, self.num_anchors, -1))
            
            
        #     if self.sample_by_seg_mask and not (ids[i] == 3):
        #         anchor_assignments = seg_mask.to(torch.int32)
        #         scale = int(self.npoints / anchor_assignments.shape[1])
        #         anchor_assignments = anchor_assignments.repeat_interleave(scale, dim=1).repeat_interleave(K, dim=0)
        #         edited_latent_per_point = gather_operation(edited_part_latent.reshape(B*K, self.num_anchors, -1).transpose(2,1).contiguous(), anchor_assignments).reshape(B*K, -1, self.npoints).transpose(1,2)
        #         edited_anchor_per_point = gather_operation(edited_anchors.reshape(B*K, self.num_anchors, -1).transpose(2, 1).contiguous(), anchor_assignments).reshape(B*K, 3, self.npoints)
        #         assert edited_anchor_per_point.shape[1] == 3
        #     else:
        #         edited_anchor_per_point = edited_anchors.reshape(B*K, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*K, self.npoints, -1).transpose(1,2)
        #         assert edited_anchor_per_point.shape[1] == 3
        #         edited_latent_per_point = edited_part_latent.reshape(B*K, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*K, self.npoints, -1)
            

        #     pred_edit = self.decode(edited_anchor_per_point, code=edited_global_feature, pointwise_latent=edited_latent_per_point, device=device, variance=None)
        #     for k in range(K):
        #         # for t in range(20, 200, 20):
        #         #     pred[f'edit_pred: "{prompts[i]}" sample {k} T={t}'] = pred_edit[t].reshape(B, K, self.npoints, 3)[:, k]
        #         pred[f'edit_pred: "{prompts[i]}" sample {k}'] = pred_edit['pred'].reshape(B, K, self.npoints, 3)[:, k]
        


    def sample(self, sample_num, fixed_id, valid_id, device, epoch, K=10):
        part_code=None
        fixed_id = torch.tensor(fixed_id).to(device)
        return self.encoder.sample_latents(sample_num, self.npoints, device, fixed_id=fixed_id, valid_id=valid_id, epoch=epoch, K=self.cimle_sample_num, part_code=part_code)
        # return self.encoder.sample_with_fixed_latents()
        # return self.encoder.sample_variation_with_fixed_latents()
        # return self.encoder.sample_with_one_fixed_latents()
        # return self.encoder.sample_with_given_latents()

    def cache_noise(self, pcds, device, eval_whole=False):
        # cache noise for CIMLE 
        from difffacto.datasets.evaluation_utils import distChamfer
        noise, id = self.encoder.sample_noise(pcds, device, self.sample_noise_num)
        if not eval_whole:
            selected_noise = gather_operation(noise.transpose(1,2).contiguous(), id.unsqueeze(1).to(torch.int32))
            return selected_noise.squeeze(-1)
        
        ref = pcds['ref'].to(device)
        seg_mask = pcds['ref_seg_mask'].to(device)
        valid_id = pcds['present'].to(device)
        B, N, C = ref.shape
        ctx, mean_per_point, logvar_per_point, flag_per_point, _ = self.encoder(pcds, device, noise=noise)
        variance_per_point = torch.exp(logvar_per_point)
        seg_mask, valid_id = map(lambda t: t.repeat_interleave(self.sample_noise_num, dim=0), [seg_mask, valid_id])
        variance_per_point = torch.exp(logvar_per_point)
        pred = self.decode(mean_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=seg_mask.to(torch.int32), valid_id=valid_id) 
        shift = pcds['shift'].to(device)
        scale = pcds['scale'].to(device)
        preds = rearrange(pred['pred'] , '(b n) ... -> b n ...', n=self.sample_noise_num) * scale.unsqueeze(1) + shift.unsqueeze(1)
        ref = ref * scale + shift
        selected_noise = []
        for i in range(ref.shape[0]):
            pred_pcd = preds[i]
            ref_pcd = ref[i].unsqueeze(0).repeat_interleave(self.sample_noise_num, dim=0)
            dl, dr = distChamfer(pred_pcd, ref_pcd)
            dist = dl.mean(1) + dr.mean(1)
            mid = dist.argmin()
            selected_noise.append(noise[i, mid])
        return torch.stack(selected_noise, dim=0)
    def cimle_forward(self, pcds,device='cuda', noise=None):
        input = pcds['input'].to(device)
        ref = pcds['ref'].to(device)
        input_seg_mask = pcds['seg_mask'].to(device)
        seg_mask = pcds['ref_seg_mask'].to(device)
        valid_id = pcds['present'].to(device)
        B, N, C = ref.shape
        if noise is None:
            noise, _ = self.encoder.sample_noise(pcds, device, 10)
            
        K = noise.shape[1]
        ctx, mean_per_point, logvar_per_point, flag_per_point, _, _ = self.encoder(pcds, device, noise=noise)
        variance_per_point = torch.exp(logvar_per_point)
        priors = torch.randn_like(variance_per_point).transpose(1,2) * torch.sqrt(variance_per_point.transpose(1,2)) + mean_per_point.transpose(1,2)
        seg_mask, valid_id = map(lambda t: t.repeat_interleave(K, dim=0), [seg_mask, valid_id])
        variance_per_point = torch.exp(logvar_per_point)
        _pred = self.decode(mean_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=seg_mask.to(torch.int32), valid_id=valid_id) 
        pred = dict()
        for i in range(K):
            for k, v in _pred.items():
                v = rearrange(v, "(b h) ... -> b h ...", h=K)
                pred[f"{k}_sample {i}"] = v[:, i]
        for i in range(K):    
            pred[f"sample prior {i}"] = priors.reshape(B, K, N, C)[:, i]
        pred['pred'] = rearrange(_pred['pred'], "(b h) ... -> b h ...", h=K)[:, 0]
        pred['input'] = input
        pred['input_ref'] = ref
        pred['seg_mask'] = input_seg_mask 
        pred['pred_seg_mask'] = seg_mask 
        pred['ref_seg_mask'] = pcds['ref_seg_mask'] 
        pred['shift'] = pcds['shift']
        pred['scale'] = pcds['scale']
        pred = {k:v.detach().cpu() for k, v in pred.items()}
        return pred
    
    def edit_latent(self, z, input, seg_flag, valid_id,  ref_means, ref_vars, fix_ids, edit_part_id, edit_part_mean, edit_part_var, fit_weight=1):
        fit_loss_dict=dict()
        part_code_means, _ = self.encoder.get_part_code(input, seg_flag)
        part_code = part_code_means.transpose(1,2)
        mean, logvar = self.encoder.get_params_from_part_code(part_code, valid_id, noise=z)
        fit_loss = F.mse_loss(torch.cat([mean, logvar], dim=1), torch.cat([ref_means, torch.log(ref_vars)], dim=1), reduction='none')
        fit_loss = fit_loss * (valid_id * fix_ids).unsqueeze(1)
        fit_loss = fit_loss.sum(dim=(-1, -2)) / (valid_id * fix_ids).sum(dim=-1)   
        if edit_part_mean is not None:
            edit_l_mean = F.mse_loss(mean[..., edit_part_id], edit_part_mean)
        else:
            edit_l_mean = torch.zeros(1).cuda()
        if edit_part_var is not None:
            edit_l_var = F.mse_loss(logvar[..., edit_part_id], torch.log(edit_part_var))
        else:
            edit_l_var = torch.zeros(1).cuda()
        edit_loss = edit_l_var + edit_l_mean
        fit_loss_dict['fit_loss'] = fit_weight * fit_loss
        fit_loss_dict['edit_loss'] = edit_loss
        if self.noise_reg_loss:
            fit_loss_dict['reg_loss'] = self.reg_loss_weight * (z ** 2).sum(1)
        return fit_loss_dict
    
    def optimize_latent(self, pcds, z, device='cuda'):
        # print(pcds.keys())
        input = pcds['input'].to(device)
        ref = pcds['ref'].to(device)
        input_seg_mask = pcds['seg_mask'].to(device)
        seg_mask = pcds['ref_seg_mask'].to(device)
        valid_id = pcds.get('present', None)
        dp_valid_id = pcds.get('dp_present', None)
        if valid_id is not None:
            valid_id = valid_id.to(device)
        if dp_valid_id is not None:
            dp_valid_id = dp_valid_id.to(device)
        loss_dict = dict()
        B, N, C = ref.shape

        ctx, mean_per_point, logvar_per_point, flag_per_point, fit_los_dict, _ = self.encoder(pcds, device, noise=z)
        if self.noise_reg_loss:
            fit_los_dict['reg_loss'] = self.reg_loss_weight * (z ** 2).sum(1)
        return fit_los_dict
    
    def pretrain(self, inputs, seg_flags, device='cuda'):
        loss_dict=dict(mse_loss=0)
        B = inputs[0].shape[0]
        if self.encoder.per_part_encoder:
            part_code_means, part_code_logvars = [], []
            for i in range(self.num_anchors):
                _m, _v = self.encoder.encoder[i](inputs[i])
                part_code_means.append(_m)
                part_code_logvars.append(_v)
            part_code_means = torch.stack(part_code_means, dim=1)
            part_code_logvars = torch.stack(part_code_logvars, dim=1)
        else:
            part_code_means, part_code_logvars = self.encoder.get_part_code(torch.cat(inputs, dim=1), torch.cat(seg_flags, dim=1))
        part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)  # (B, F, M)
        prior_dict = self.encoder.get_prior_loss(part_code, part_code_means, part_code_logvars, torch.ones([B, self.num_anchors]).cuda())
        loss_dict.update(prior_dict)
        for i in range(self.num_anchors):
            ctx = [part_code[..., i].unsqueeze(1)]
            input = inputs[i]
            t, _ = self.sampler.sample(B, device)
            diffusion_loss = self.diffusion[i].training_losses(input.transpose(1,2), t, ctx=ctx)
            loss_dict['mse_loss'] = diffusion_loss['mse_loss'] + loss_dict['mse_loss']
        loss_dict['mse_loss'] = loss_dict['mse_loss'] / self.num_anchors
        return loss_dict
    def pretrain_part(self, input, seg_flag, part_id, device='cuda'):
        loss_dict=dict()
        B = input.shape[0]
        part_code_means, part_code_logvars = self.encoder.encoder[part_id](input)
        part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars)  # (B, F)
        prior_dict = self.encoder.get_prior_loss_part( part_code, part_code_means, part_code_logvars, part_id)
        loss_dict.update(prior_dict)
        t, _ = self.sampler.sample(B, device)
        diffusion_loss = self.diffusion[part_id].training_losses(input.transpose(1,2), t, ctx=[part_code.unsqueeze(1)])
        loss_dict.update(diffusion_loss)
        return loss_dict
    
    def pretrain_validate(self, sample_num, num_points, device='cuda'):
        part_codes = torch.randn(sample_num, self.encoder.zdim, self.num_anchors).cuda()
        finals = []
        for i in range(self.num_anchors):
            final = dict()
            ctx = [part_codes[..., i]]
            for t, sample in self.diffusion[i].p_sample_loop_progressive(
                [sample_num, 3, num_points],
                anchors=torch.zeros([sample_num, 3, num_points]).cuda(),
                variance=torch.ones([sample_num, 3, num_points]).cuda(),
                ctx=[part_codes[..., i].unsqueeze(1)],
                noise=None,
                device=device,
                progress=False,
            ):
                if t == 0:
                    finals.append(sample['sample'].transpose(2, 1))
        return finals
        
    def forward(self, pcds, device='cuda', epoch=0, **kwargs):
        """
            input: [B, npoints, 3]
            attn_map: [B, npoints, n_class]
            seg_mask: [B, npoints]
        """
        if self.save_weights:
            self.eval()
            torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, "encoder_ckpt.pth"))
            torch.save(self.decomposer.state_dict(), os.path.join(self.save_dir, "decomposer_ckpt.pth"))
            exit()
        
        input = pcds['input'].to(device)
        ref = pcds['ref'].to(device)
        input_seg_mask = pcds['seg_mask'].to(device)
        seg_mask = pcds['ref_seg_mask'].to(device)
        valid_id = pcds.get('present', None)
        dp_valid_id = pcds.get('dp_present', None)
        if valid_id is not None:
            valid_id = valid_id.to(device)
        if dp_valid_id is not None:
            dp_valid_id = dp_valid_id.to(device)
        loss_dict = dict()
        B, N, C = ref.shape

        ctx, mean_per_point, logvar_per_point, flag_per_point, fit_los_dict, part_code = self.encoder(pcds, device, epoch=epoch)
        if self.zero_anchors:
            mean_per_point = torch.zeros_like(mean_per_point)
        if self.npoints < N:
            mean_per_point, logvar_per_point  = map(lambda r: r[..., :self.npoints], [mean_per_point, logvar_per_point])
            ctx = [r[..., self.npoints] if r.shape[-1] > self.npoints else r for r in ctx]
            ref = ref[:, :self.npoints]
            seg_mask = seg_mask[:, :self.npoints]
        variance_per_point = torch.exp(logvar_per_point)
        if self.training:
            
            t, _ = self.sampler.sample(B, device)
            

            loss_dict.update(fit_los_dict)

            if self.detach_anchor:
                mean_per_point = mean_per_point.detach()
            if self.detach_variance:
                logvar_per_point = logvar_per_point.detach()
            
            if dp_valid_id is not None:
                dp_flags = gather_operation(dp_valid_id.unsqueeze(1).contiguous(), seg_mask.to(torch.int32)).reshape(B, 1, self.npoints) 
            else:
                dp_flags = None
            diffusion_loss = self.diffusion.training_losses(ref.transpose(1,2) if not self.use_input else input.transpose(1,2), t, anchors=mean_per_point, variance=variance_per_point, ctx=ctx, anchor_assignment=seg_mask.to(torch.int32), valid_id=dp_valid_id, flags=dp_flags)
            diffusion_loss['mse_loss'] = self.diffusion_loss_weight * diffusion_loss['mse_loss']
            loss_dict.update(diffusion_loss)
            return loss_dict

        else:
            
            if self.interpolate and not self.training:
                return [(self.interpolate_latent(device, pcds), "interpolate")]
            if self.combine and not self.training:
                return [(self.combine_latent(pcds, device), "mixing")]
            if self.drift_anchors:
                return [(self.interpolate_params(device, pcds), "interpolate_params")] 
            
            if not self.training and self.gen:
                # fixed_ids = [[0,0,0,0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 1]]
                fixed_ids = [0, 0, 0, 0]
                out_list = []
                if self.fix_part_ids is not None:
                    for i in self.fix_part_ids:
                        fixed_ids[i] = 1
                for fixed_id in [fixed_ids]:
                    ctx, mean_per_point, logvar_per_point, _seg_mask, _valid_id, latents = self.sample(B, fixed_id, valid_id, device, epoch, K=10)
                    fixed_codes, min_noises, min_means, min_logvars = latents
                    variance_per_point = torch.exp(logvar_per_point)
                    _pred = self.decode(mean_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=_seg_mask.to(torch.int32), valid_id=_valid_id) 
                    # pred[f"sample prior"] = priors
                    # _pred['pred_seg_mask'] = _seg_mask 
                    # _pred['means'] = min_means
                    # _pred['logvars'] = min_logvars
                    # _pred['noises'] = min_noises
                    # _pred['part_codes'] = fixed_codes
                    # return [(_pred,  "car_chosen_variation")]
                    pred = dict()
                    if self.cimle:
                        priors = torch.randn_like(variance_per_point.transpose(1,2)) * torch.sqrt(variance_per_point.transpose(1,2)) + mean_per_point.transpose(1,2)
                        if self.use_input:
                            _pred['pred'] = _pred['pred'] * torch.sqrt(variance_per_point.transpose(1,2)) + mean_per_point.transpose(1,2)
                        for i in range(self.cimle_sample_num):
                            for k, v in _pred.items():
                                v = rearrange(v, "(b h) ... -> b h ...", h=self.cimle_sample_num)
                                pred[f"{k}_sample {i}"] = v[:, i]
                        for i in range(self.cimle_sample_num):    
                            pred[f"sample prior {i}"] = priors.reshape(B, self.cimle_sample_num, self.npoints, C)[:, i]
                        pred['pred'] = rearrange(_pred['pred'], "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0]
                        pred['pred_seg_mask'] = rearrange(_seg_mask, "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0]
                        pred['anchors'] = rearrange(mean_per_point, "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0].transpose(1, 2)
                    else:
                        pred = _pred
                        priors = torch.randn_like(variance_per_point.transpose(1,2)) *  torch.sqrt(variance_per_point.transpose(1,2)) + mean_per_point.transpose(1,2)
                        pred[f"sample prior"] = priors
                        pred['pred_seg_mask'] = _seg_mask 
                        pred['anchors'] = mean_per_point.transpose(1, 2)
                        
                    pred['input'] = input
                    pred['input_ref'] = ref
                    pred['ref_seg_mask'] = pcds['ref_seg_mask']
                    pred['seg_mask'] = input_seg_mask 
                    
                    pred['present'] = valid_id
                    pred['shift'] = pcds['shift']
                    pred['scale'] = pcds['scale']
                    pred = {k:v.detach().cpu() for k, v in pred.items()}
                    out_list.append((pred,  f"gen_fixed" + "".join(map(lambda i: str(i), fixed_id))))
                return out_list
            elif self.cimle:
                noise, _ = self.encoder.sample_noise(pcds, device, self.cimle_sample_num)
                ctx, mean_per_point, logvar_per_point, _, _, latents = self.encoder(pcds, device, noise=noise)
                part_code, mean, logvar, noise = latents
                seg_mask, valid_id = map(lambda t: t.repeat_interleave(self.cimle_sample_num, dim=0), [seg_mask, valid_id])
                variance_per_point = torch.exp(logvar_per_point)
            N = mean_per_point.shape[-1]
            if self.npoints > mean_per_point.shape[-1]:
                mean_per_point, variance_per_point = mean_per_point.repeat_interleave(self.npoints // N, dim=-1), variance_per_point.repeat_interleave(self.npoints // N, dim=-1)
                seg_mask = seg_mask.repeat_interleave(self.npoints // N, dim=-1)
            _pred = self.decode(mean_per_point, ctx=ctx, device=device, variance=variance_per_point, anchor_assignments=seg_mask.to(torch.int32), valid_id=valid_id) 
            pred = dict()
            if self.cimle:
                for i in range(self.cimle_sample_num):
                    for k, v in _pred.items():
                        v = rearrange(v, "(b h) ... -> b h ...", h=self.cimle_sample_num)
                        pred[f"{k}_sample {i}"] = v[:, i]
                for i in range(self.cimle_sample_num):    
                    priors = torch.randn_like(variance_per_point).transpose(1,2) *  torch.sqrt(variance_per_point.transpose(1,2)) + mean_per_point.transpose(1,2)
                    pred[f"sample prior {i}"] = priors.reshape(B, self.cimle_sample_num, self.npoints, C)[:, i]
                    pred[f"noise latent {i}"] = noise.reshape(B, self.cimle_sample_num, 32)[:, i]
                    pred[f"sample {i} mean"] = mean.reshape(B, self.cimle_sample_num, 3, self.num_anchors)[:, i]
                    pred[f"sample {i} logvar"] = logvar.reshape(B, self.cimle_sample_num, 3, self.num_anchors)[:, i]
                pred['pred'] = rearrange(_pred['pred'], "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0]
                pred['pred_seg_mask'] = rearrange(seg_mask, "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0]
                pred['anchors'] = rearrange(mean_per_point, "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0].transpose(1, 2)
                pred['part_latents'] = rearrange(part_code, "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0]
                pred['valid_id'] = rearrange(valid_id, "(b h) ... -> b h ...", h=self.cimle_sample_num)[:, 0]
            else:
                pred = _pred
                pred['pred_seg_mask'] = seg_mask 
                pred['anchors'] = mean_per_point.transpose(1, 2)
                priors = torch.randn_like(variance_per_point.transpose(1,2)) * torch.sqrt(variance_per_point.transpose(1,2)) + mean_per_point.transpose(1,2)
                pred[f"sample prior"] = priors
            # import ipdb; ipdb.set_trace()
            pred['input'] = input
            pred['input_ref'] = ref
            pred['ref_seg_mask'] = pcds['ref_seg_mask']
            pred['seg_mask'] = input_seg_mask 
            pred['token'] = pcds['token']
            pred['present'] = valid_id
            pred['shift'] = pcds['shift']
            pred['scale'] = pcds['scale']
            pred = {k:v.detach().cpu()  if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
            
            out_list = [(pred, "sample")]
            # out_list.append((self.interpolate_latent(device, pcds), "interpolate"))
            # out_list.append((self.combine_latent(pcds, device), "mixing"))
            # out_list.append((self.interpolate_params(device, pcds), "interpolate_params"))
            return out_list
