import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Module, functional
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS, DIFFUSIONS, SAMPLERS, METRICS, DECOMPOSERS, SEGMENTORS
from pointnet2_ops.pointnet2_utils import gather_operation
from .language_utils.language_util import tokenizing
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
class AnchorDiffGenSuperSegments(Module):

    def __init__(
        self, 
        encoder, 
        decomposer,
        diffusion, 
        sampler, 
        num_anchors,
        num_timesteps, 
        npoints=2048, 
        anchor_loss_weight=1.,
        anchor_weight_annealing=False,
        loss=None,
        completion=False,
        include_attn_weight_in_encoder=True,
        contrastive_loss=None,
        contrastive_weight=1.,
        detach_anchor=False,
        guidance=False,
        part_latent_dropout_prob=0.2,
        global_latent_dropout_prob=0.2,
        language_edit=False,
        language_encoder=None,
        latent_language_fuser=None,
        language_encoder_ckpt=None,
        update_mlp_ckpt=None,
        latent_diffusion=None,
        ldm_ckpt=None,
        use_primary=False,
        use_zero_anchor=False,
        use_global_anchor=False,
        use_gt_anchors=False,
        project_latent=False,
        project_pe_type = 0,
        post_ff=False,
        post_norm='gn',
        post_dp=0.2,
        use_log_for_scale=True,
        share_projection=True,
        learn_var=False,
        detach_variance=True,
        part_dim=256,
        global_shift=False,
        global_scale=False,
        vertical_only=True,

        sample_by_seg_mask=False,
        ret_traj=False, 
        ret_interval=20, 
        forward_sample=False,
        interpolate=False,
        combine=False,
        drift_anchors=False,
        save_pred_xstart=False,
        partglot_dataset=False,
        save_dir=None,
        save_weights=False,
        normal_diffusion=False,
        annealing_epoch=500,
        intervaled_training=False,
        cache_interval=100,
        freeze_interval=50,
        freeze_diffusion=False,
        freeze_encoder=False
        ):
        super().__init__()
        self.encoder = build_from_cfg(encoder, ENCODERS, num_anchors=num_anchors)
        self.diffusion = build_from_cfg(diffusion, DIFFUSIONS, num_timesteps=num_timesteps)
        self.decomposer = build_from_cfg(decomposer, DECOMPOSERS, num_anchors=num_anchors, point_dim=3)
        self.sampler = build_from_cfg(sampler, SAMPLERS, num_timesteps=num_timesteps)
        self.loss_func=build_from_cfg(loss, METRICS)
        self.use_zero_anchor=use_zero_anchor
        self.use_global_anchor=use_global_anchor
        self.use_gt_anchors=use_gt_anchors
        self.intervaled_training=intervaled_training
        self.cache_interval=cache_interval
        self.freeze_interval=freeze_interval
        self.freeze_diffusion=freeze_diffusion
        self.freeze_encoder=freeze_encoder
        if cache_interval is not None and freeze_interval is not None:
            assert cache_interval > freeze_interval
        self.normal_diffusion=normal_diffusion
        self.learn_var = learn_var
        param_dim = 6 if learn_var else 3
        self.param_dim=param_dim
        if latent_diffusion is not None:
            self.latent_diffusion = build_from_cfg(latent_diffusion, DIFFUSIONS, num_part=num_anchors)
            self.latent_diffusion.load_state_dict(torch.load(ldm_ckpt))
        self.save_weights=save_weights
        self.save_dir=save_dir
        self.completion = completion
        self.global_shift = global_shift
        self.global_scale=global_scale
        self.vertical_only=vertical_only
        self.anchor_weight_annealing=anchor_weight_annealing
        if anchor_weight_annealing:
            self.start_weight=0.
            self.end_weight=1.
            self.ratio = (self.end_weight - self.start_weight) / annealing_epoch
            self.annealing_epoch=annealing_epoch
        self.drift_anchors=drift_anchors
        self.sample_by_seg_mask=sample_by_seg_mask
        self.use_log_for_scale=use_log_for_scale
        self.include_attn_weight_in_encoder=include_attn_weight_in_encoder
        self.num_timesteps = int(num_timesteps)
        self.num_anchors = num_anchors
        self.detach_anchor=detach_anchor
        self.detach_variance=detach_variance
        self.use_primary=use_primary
        self.language_edit=language_edit
        self.project_latent=project_latent 
        self.part_dim=part_dim
        if project_latent:
            self.project_pe_type = project_pe_type
            self.post_ff = post_ff 
            if post_norm == 'bn':
                post_norm = nn.BatchNorm1d
            elif post_norm == 'gn':
                post_norm = Normalize
            elif post_norm is None:
                post_norm = identity
            self.share_projection=share_projection
            if share_projection:
                if self.project_pe_type == 0:
                    self.down_proj = nn.Linear(part_dim, part_dim + param_dim)
                    self.up_proj = nn.Linear(param_dim, self.part_dim)
                elif self.project_pe_type == 1:
                    self.down_proj = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Linear(part_dim, 128 ),
                        nn.ReLU(inplace=True),
                        nn.Linear(128 , 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64 , param_dim),
                    )
                    self.up_proj = nn.Linear(param_dim, self.part_dim)
                elif self.project_pe_type == 2:
                    self.down_proj = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Linear(part_dim * self.num_anchors, 256 ),
                        nn.ReLU(inplace=True),
                        nn.Linear(256 , 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64 , param_dim * self.num_anchors),
                    )
                    self.up_proj = nn.Linear(param_dim, self.part_dim)
            else:
                self.down_proj = nn.Conv1d(part_dim * self.num_anchors, (part_dim + param_dim) * self.num_anchors, 1, groups=self.num_anchors)
                self.up_proj = nn.Conv1d(param_dim * self.num_anchors, part_dim * self.num_anchors,  1, groups=self.num_anchors)
            if self.post_ff:
                self.out_layers = nn.Sequential(
                    post_norm(part_dim),
                    nn.SiLU(),
                    nn.Dropout(p=post_dp),
                    zero_module(
                        nn.Conv1d(part_dim, part_dim, 1)
                    ),
                )
        self.partglot_dataset=partglot_dataset
        if self.language_edit:
            self.language_encoder = build_from_cfg(language_encoder, ENCODERS, vocab_size=VOCAB_SIZE)
            self.latent_language_encoder = build_from_cfg(latent_language_fuser, ENCODERS, num_part=num_anchors)
            self.language_encoder.load_state_dict(torch.load(language_encoder_ckpt))
            if self.latent_language_encoder is not None:
                self.latent_language_encoder.load_state_dict(torch.load(update_mlp_ckpt))
                self.icmle = self.latent_language_encoder.conditional
                self.conditional_dim=self.latent_language_encoder.conditional_dim
        self.npoints = npoints
        self.points_per_anchor = self.npoints // self.num_anchors
        self.forward_sample = forward_sample
        self.anchor_loss_weight=anchor_loss_weight
        self.interpolate=interpolate
        self.ret_traj=ret_traj
        self.ret_interval=ret_interval
        self.save_pred_xstart=save_pred_xstart
        self.combine=combine
        self.contrastive_loss=build_from_cfg(contrastive_loss, METRICS)
        self.contrastive_weight=contrastive_weight
        self.guidance=guidance
        self.part_latent_dropout_prob = part_latent_dropout_prob
        self.global_latent_dropout_prob=global_latent_dropout_prob
        print(self)

    
    def decode(self, anchors, code=None, pointwise_latent=None, noise=None, variance=None, anchor_assignments=None, device='cuda'):
        final = dict() 
        bs = anchors.shape[0]
        for t, sample in self.diffusion.p_sample_loop_progressive(
            [bs, 3, self.npoints],
            anchors=anchors,
            code=code,
            variance=variance,
            pointwise_latent=pointwise_latent,
            noise=noise,
            anchor_assignment=anchor_assignments,
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

    def interpolate_latent(self, seg_mask, device, pcds):
        # ids = [(19,20, 2), (17, 15, 0), (37, 38, 1), (122, 120, 1), (19, 18, 2), (66,65, 2), (1, 0, 2)]
        ids = [(0,1, 1), (6, 8, 2), (12, 11, 1), (11, 12, 2), (40, 38, 2), (51,50, 1), (52, 53, 2)]
        out_dict = {f"interpolate sample {i}":[] for i in range(10)}
        # out_dict['combined'] = []
        # out_dict['seg_mask_ref'] = []
        out_dict['input1_ref'] = []
        out_dict['input2_ref'] = []
        out_dict['seg_mask1_ref'] = []
        out_dict['seg_mask2_ref'] = []
        out_dict['seg_mask'] = []
        out_dict.update({f"interpolate anchor sample {i}": [] for i in range(10)})
        latents = self.get_primary_latent(pcds['input'].to(device), attn=pcds['attn_map'].to(device))
        for id1, id2, anchor_id in ids:
            latent1, latent2 = latents[id1], latents[id2]

            dx = torch.zeros([10, self.num_anchors, 1]).to(device)
            dx[:, anchor_id] += torch.linspace(0, 1, steps=10).reshape(-1, 1).to(device)
            primary_latent = latent1 + (latent2 - latent1) * dx
            B = primary_latent.shape[0]
            if self.project_latent:
                primary_latent, shift, scale, _ = self.project_primary_latent(primary_latent, interpolate=True)
            global_feat, new_anchors, secondary_latent = self.decomposer(primary_latent)
            if self.project_latent:
                new_anchors = shift 
                new_vars = scale
            else:
                new_vars = None
            if self.use_primary:
                new_latents = primary_latent 
            else:
                new_latents = secondary_latent
            if self.use_zero_anchor:
                new_anchors = torch.zeros_like(new_anchors)
            if self.use_global_anchor:
                new_anchors = pcds['global_anchor_mean'].cuda()
            if self.use_gt_anchors:
                new_anchors = pcds['part_means'].cuda()[..., 1:]
            if self.sample_by_seg_mask:
                anchor_assignments = seg_mask[id1].to(torch.int32).unsqueeze(0).repeat_interleave(B, dim=0)
                multiple = int(self.npoints / anchor_assignments.shape[1])
                anchor_assignments = anchor_assignments.repeat_interleave(multiple, dim=1)
                latent_per_point, anchor_per_point, variance_per_point = self.gather_all(anchor_assignments, new_latents, new_anchors, new_vars)
            else:
                latent_per_point = new_latents.reshape(10, self.num_anchors, 1, -1).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, -1)
                anchor_per_point = new_anchors.reshape(10, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, 3).transpose(1,2)
                anchor_assignments = torch.arange(self.num_anchors).to(new_anchors.device).reshape(1, -1, 1).expand(10, -1, self.points_per_anchor).reshape(10, self.npoints)
                if self.learn_var:
                    variance_per_point = new_vars.reshape(10, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, 3)
                else:
                    variance_per_point=None
            if self.normal_diffusion:
                anchor_per_point = anchor_per_point * 0
                variance_per_point = torch.ones_like(variance_per_point)

            pred = self.decode(anchor_per_point, code=global_feat, pointwise_latent=latent_per_point, device=device, variance=variance_per_point, anchor_assignments=anchor_assignments)
            # ref = torch.zeros([self.npoints, 3]).cuda()
            # _ref = torch.cat([gt1[seg_mask[id1] != anchor_id, :3], gt2[seg_mask[id2] == anchor_id, :3]], dim=0)
            # ref[:_ref.shape[0]] = _ref
            # seg_mask_ref = torch.zeros_like(seg_mask[0])
            # _seg_mask_ref = torch.cat([seg_mask[id1][seg_mask[id1] != anchor_id], seg_mask[id2][seg_mask[id2] == anchor_id]], dim=0)
            # seg_mask_ref[:_seg_mask_ref.shape[0]] = _seg_mask_ref
            for i in range(pred['pred'].shape[0]):
                out_dict[f"interpolate sample {i}"].append(pred['pred'][i])
                if pred.get(self.num_timesteps, None) is not None:
                    if f"interpolate sample {i} priors" not in out_dict.keys():
                        out_dict[f"interpolate sample {i} priors"] = []
                    out_dict[f"interpolate sample {i} priors"].append(pred[self.num_timesteps][i])
                out_dict[f"interpolate anchor sample {i}"].append(new_anchors[i])
            # out_dict['combined'].append(ref)
            # out_dict['seg_mask_ref'].append(seg_mask_ref)
            out_dict['input1_ref'].append(pcds['input'][id1, :, 0:3])
            out_dict['input2_ref'].append(pcds['input'][id2, :, 0:3])
            out_dict['seg_mask1_ref'].append(seg_mask[id1])
            out_dict['seg_mask'].append(seg_mask[id1])
            out_dict['seg_mask2_ref'].append(seg_mask[id2])
        out_dict = {k:torch.stack(v, dim=0) for k, v in out_dict.items()}
        out_dict['pred'] = pcds['input'][0].unsqueeze(0).repeat_interleave(len(ids), dim=0)
        out_dict['input_ref'] = pcds['input'][0].unsqueeze(0).repeat_interleave(len(ids), dim=0)
        out_dict['shift'] = pcds['shift'][0].unsqueeze(0)
        out_dict['scale'] = pcds['scale'][0].unsqueeze(0)
        return out_dict

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

    def combine_latent(self, x, pcds, seg_mask, ids, device):
        idss = [(18, 21, 12, 19)]
        primary_latents = self.get_primary_latent(pcds['input'].to(device), attn=pcds['attn_map'].to(device))
        for ids in idss:
            primary_latents = torch.stack([primary_latents[i] for i in ids], dim=0)
            primary_latents = torch.stack([primary_latents[i, i] for i in range(self.num_anchors)], dim=0).reshape(1, self.num_anchors, -1)
            if self.project_latent:
                primary_latents, shift, scale, _ = self.project_primary_latent(primary_latents, mixing=True)
            global_feature, new_anchors, secondary_latent = self.decomposer(primary_latents)
            if self.project_latent:
                new_anchors = shift 
                new_vars = scale
            else:
                new_vars = None
            if self.use_primary:
                new_latents = primary_latents
            else:
                new_latents = secondary_latent
            if self.use_zero_anchor:
                new_anchors = torch.zeros_like(new_anchors)
            if self.use_global_anchor:
                new_anchors = pcds['global_anchor_mean'].cuda()
            if self.use_gt_anchors:
                new_anchors = torch.stack([pcds['part_means'].cuda()[ids[i], i + 1] for i in range(self.num_anchors)], dim=0)
            latent_per_point = new_latents.reshape(1, self.num_anchors, 1, -1).expand(-1, -1, self.points_per_anchor, -1).reshape(1, self.npoints, -1)
            param_per_point = new_params.reshape(1, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(1, self.npoints, 3).transpose(1,2)
            assert param_per_point.shape[1] == 3
            anchor_per_point = param_per_point
            variance_per_point = None
        
        pred = self.decode(anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, device=device, variance=variance_per_point) 
        ref = torch.cat([gts[i, seg_mask[ids[i]] == i, :3] for i in range(self.num_anchors)], dim=0)[:self.npoints]
        seg_mask_ref = torch.cat([seg_mask[ids[i]][seg_mask[ids[i]] == i] for i in range(self.num_anchors)], dim=0)
        print(ref.shape)
        pred['input_ref'] = ref.unsqueeze(0)
        pred['seg_mask_ref'] = seg_mask_ref.unsqueeze(0)
        for i in range(self.num_anchors):
            pred[f'input_{i}'] = gts[i, :, 0:3].unsqueeze(0)
            pred[f'seg_mask_{i}'] = seg_mask[ids[i]].unsqueeze(0)
        pred['anchors'] = new_params[..., :3]
        pred['shift'] = pcds['shift']
        pred['scale'] = pcds['scale']
        return pred

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
        if self.partglot_dataset:
            target_pcds = data['target'].to(device)
            target_attn_maps = data['target_attn_map'].to(device)
            distractor_pcds = data['distractor'].to(device)
            distractor_attn_maps = data['distractor_attn_map'].to(device)
            part_indicators = data['part_indicator'].to(device).repeat_interleave(3, dim=0)
            texts = data['text'].to(device)
            B, npoint = target_pcds.shape[:-1]
            pcds = torch.stack([target_pcds, distractor_pcds, distractor_pcds], dim=1).reshape(B*3, npoint, 3)
            attn_maps = torch.stack([target_attn_maps, distractor_attn_maps, distractor_attn_maps], dim=1).reshape(B*3, npoint, -1)
            input = torch.cat([pcds, attn_maps], dim=-1)
            language_enc_f, language_enc_attn = self.language_encoder(texts)
        
        
            part_latent = self.encoder(input)
            n_part, n_dim = part_latent.shape[1:]
            

            part_id = part_indicators.max(1)[1].to(torch.int)
            referenced_part_latent = gather_operation(part_latent.transpose(1,2).contiguous(), part_id.reshape(B*3, 1)).reshape(B, 3, 1, -1)
            tgt_latent, latent_for_edit, distractor_latent = referenced_part_latent.reshape(B, 3, 1, n_dim).split(1, dim=1)
            tgt_latent, latent_for_edit, distractor_latent = tgt_latent.reshape(B, n_dim), latent_for_edit.reshape(B, n_dim), distractor_latent.reshape(B, n_dim)
            modified_part_latent_res = self.update_mlp(torch.cat([language_enc_f, latent_for_edit], -1))
            modified_part_latent = latent_for_edit + modified_part_latent_res
            modified_part_latent = torch.stack([tgt_latent, modified_part_latent, distractor_latent], dim=1).reshape(B*3, 1, n_dim)
            total_modified_part_latent = (1 - part_indicators).reshape(B*3, n_part, 1) * part_latent + part_indicators.reshape(B*3, n_part, 1) * modified_part_latent
            global_feature, anchors, part_latent = self.decomposer(total_modified_part_latent)
            
            
            assert anchors.shape[1] == part_latent.shape[1] == self.num_anchors
            param_per_point = anchors.reshape(B*3, self.num_anchors, 1, 3).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*3, self.npoints, 3).transpose(1,2)
            assert param_per_point.shape[1] == 3
            anchor_per_point = param_per_point

            latent_per_point = part_latent.reshape(B*3, self.num_anchors, 1, n_dim).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*3, self.npoints, n_dim)
            pred = self.decode(anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, device=device, variance=None) 
            pred_out = dict()
            for k, v in pred.items():
                target, edit, distractor = v.reshape(B, 3, self.npoints, 3).split(1, dim=1)
                pred_out[f"distractor_{k}"] = distractor.squeeze(1)
                pred_out[f"distractor_edit_{k}"] = edit.squeeze(1)
                pred_out[f"target_{k}"] = target.squeeze(1)
            

            pred_out['text'] = texts
            pred_out['input_distractor'] = distractor_pcds
            pred_out['input_target'] = target_pcds
            pred_out['anchors'] = anchors
            pred_out['seg_mask'] = distractor_attn_maps.max(-1)[1]
            pred_out['seg_mask_ref'] = target_attn_maps.max(-1)[1]
            pred_out['assigned_anchor'] = anchor_per_point.transpose(1, 2)
            pred_out['target_shift'] = data['target_shift']
            pred_out['target_scale'] = data['target_scale']
            pred_out['distractor_shift'] = data['distractor_shift']
            pred_out['distractor_scale'] = data['distractor_scale']
            return pred_out
        else:
            input = data['input'].to(device)
            ref = data['ref'].to(device)
            attn_map = data['attn_map'].to(device) # ->[B, npoints, n_class]
            seg_mask = data['seg_mask'].to(device)
            B = input.shape[0]
            x = torch.cat([input, attn_map], dim=-1).repeat_interleave(2, dim=0) # -> [B, npoints, n_class + 3]
            part_latent = self.encoder(x)
            input_part_latent, part_latent_for_edit = part_latent.reshape(B, 2, self.num_anchors, -1).split(1, dim=1)
            input_part_latent, part_latent_for_edit = input_part_latent.reshape(B, self.num_anchors, -1), part_latent_for_edit.reshape(B, self.num_anchors, -1)
            input_global_feature, input_anchor, input_part_latent = self.decomposer(input_part_latent)

            if self.sample_by_seg_mask:
                anchor_assignments = seg_mask.to(torch.int32)
                scale = int(self.npoints / anchor_assignments.shape[1])
                anchor_assignments = anchor_assignments.repeat_interleave(scale, dim=1)
                input_latent_per_point = gather_operation(input_part_latent.transpose(2,1).contiguous(), anchor_assignments).reshape(B, -1, self.npoints).transpose(1,2)
                input_anchor_per_point = gather_operation(input_anchor.transpose(2, 1).contiguous(), anchor_assignments).reshape(B, 3, self.npoints)
                assert input_anchor_per_point.shape[1] == 3
            else:
                input_anchor_per_point = input_anchor.reshape(B, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, -1).transpose(1,2)
                assert input_anchor_per_point.shape[1] == 3
                input_latent_per_point = input_part_latent.reshape(B, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, -1)
            
            pred = self.decode(input_anchor_per_point, code=input_global_feature, pointwise_latent=input_latent_per_point, device=device, variance=None) 
            pred['input'] = input
            pred['input_ref'] = ref
            pred['anchors'] = input_anchor
            pred['seg_mask'] = seg_mask 
            pred['seg_mask_ref'] = seg_mask 
            pred['assigned_anchor'] = input_anchor_per_point.transpose(1, 2)
            pred['shift'] = data['shift']
            pred['scale'] = data['scale']

            if self.latent_diffusion is not None:
                part_latent_for_edit = part_latent_for_edit.transpose(1,2)
                input_anchor = input_anchor.transpose(1,2)
            for i, prompt in enumerate(tokenized_prompts):
                print(prompts[i])
                prompt = prompt.to(device).unsqueeze(0)
                one_hot_id = torch.eye(self.num_anchors).to(device)[ids[i]].unsqueeze(0).repeat_interleave(B, dim=0)
                language_enc_f, language_enc_attn = self.language_encoder(prompt)
                language_enc_f = language_enc_f.repeat_interleave(B, dim=0)
                if self.latent_diffusion is not None:
                    
                    language_enc_f = language_enc_f.transpose(1,2)
                    # print(part_latent_for_edit.shape, language_enc_f.shape)
                    total_modified_part_latent = self.latent_diffusion.p_sample_loop(part_latent_for_edit.shape, part_latent_for_edit, language_enc_f, anchors=input_anchor)
                    # print(total_modified_part_latent.shape)
                    total_modified_part_latent = total_modified_part_latent.transpose(1, 2).unsqueeze(1)
                    K = total_modified_part_latent.shape[1]
                    # total_modified_part_latent = part_latent_for_edit + torch.rand_like(part_latent_for_edit)
                else:
                    referenced_part_latent = part_latent_for_edit[:, ids[i]]
                    if self.icmle:
                        K=10
                        conditional = torch.randn([B, K, self.conditional_dim]).cuda()
                    else:
                        K=1
                        conditional = None
                    total_modified_part_latent = self.latent_language_encoder(part_latent_for_edit, one_hot_id, language_enc_f, conditional=conditional)
                edited_global_feature, edited_anchors, edited_part_latent = self.decomposer(total_modified_part_latent.reshape(B*K, self.num_anchors, -1))
                
                
                if self.sample_by_seg_mask and not (ids[i] == 3):
                    anchor_assignments = seg_mask.to(torch.int32)
                    scale = int(self.npoints / anchor_assignments.shape[1])
                    anchor_assignments = anchor_assignments.repeat_interleave(scale, dim=1).repeat_interleave(K, dim=0)
                    edited_latent_per_point = gather_operation(edited_part_latent.reshape(B*K, self.num_anchors, -1).transpose(2,1).contiguous(), anchor_assignments).reshape(B*K, -1, self.npoints).transpose(1,2)
                    edited_anchor_per_point = gather_operation(edited_anchors.reshape(B*K, self.num_anchors, -1).transpose(2, 1).contiguous(), anchor_assignments).reshape(B*K, 3, self.npoints)
                    assert edited_anchor_per_point.shape[1] == 3
                else:
                    edited_anchor_per_point = edited_anchors.reshape(B*K, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*K, self.npoints, -1).transpose(1,2)
                    assert edited_anchor_per_point.shape[1] == 3
                    edited_latent_per_point = edited_part_latent.reshape(B*K, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B*K, self.npoints, -1)
                

                pred_edit = self.decode(edited_anchor_per_point, code=edited_global_feature, pointwise_latent=edited_latent_per_point, device=device, variance=None)
                for k in range(K):
                    # for t in range(20, 200, 20):
                    #     pred[f'edit_pred: "{prompts[i]}" sample {k} T={t}'] = pred_edit[t].reshape(B, K, self.npoints, 3)[:, k]
                    pred[f'edit_pred: "{prompts[i]}" sample {k}'] = pred_edit['pred'].reshape(B, K, self.npoints, 3)[:, k]
        return pred
        
    def anchor_drift(self, pcds, x, global_feature, anchors, part_latent, seg_mask, device):
        out_dict = dict()
        B = global_feature.shape[0]
        direction = F.normalize(torch.randn(B, 10, self.num_anchors, 1).cuda())
        part_id = torch.tensor([1.,1., 1., 1.]).to(float).cuda().reshape(1, 1, -1, 1)
        scales = torch.linspace(0., 2, 10).to(float).cuda().reshape(1, -1, 1, 1)
        part_latent = part_latent.repeat_interleave(10, dim=0)
        global_feature = global_feature.repeat_interleave(10, dim=0)
        if self.project_latent:
            primary_part_latent = self.encoder(x)
            intrinsic, anchors = torch.split(self.down_proj(primary_part_latent), [self.part_dim, 3], dim=-1)
            intrinsic = intrinsic.repeat_interleave(10, dim=0)
        shifted_anchors = direction * scales  * part_id + anchors.unsqueeze(1) * part_id #+ anchors.unsqueeze(1)
        shifted_anchors = shifted_anchors.reshape(B*10, self.num_anchors, 3).to(torch.float32)
        if self.project_latent:
            primary_part_latent = intrinsic + self.up_proj(shifted_anchors)
            global_feature, _, part_latent = self.decomposer(primary_part_latent)
        if self.use_primary:
            part_latent = primary_part_latent 
        
        anchor_assignments = seg_mask.to(torch.int32).repeat_interleave(10, dim=0)
        scale = int(self.npoints / anchor_assignments.shape[1])
        anchor_assignments = anchor_assignments.repeat_interleave(scale, dim=1)
        latent_per_point = gather_operation(part_latent.transpose(2,1).contiguous(), anchor_assignments).reshape(B*10, -1, self.npoints).transpose(1,2)
        anchor_per_point = gather_operation(shifted_anchors.transpose(2, 1).contiguous(), anchor_assignments).reshape(B*10, 3, self.npoints)
        print(latent_per_point.shape, anchor_per_point.shape)
        assert anchor_per_point.shape[1] == 3
        pred = self.decode(anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, device=device, variance=None) 
        for k in range(10):
            # for t in range(20, 200, 20):
            #     pred[f'edit_pred: "{prompts[i]}" sample {k} T={t}'] = pred_edit[t].reshape(B, K, self.npoints, 3)[:, k]
            out_dict[f'anchor sample {k}'] = shifted_anchors.reshape(B, 10, self.num_anchors, 3)[:, k]
            out_dict[f'pred anchor drift sample {k}'] = pred['pred'].reshape(B, 10, self.npoints, 3)[:, k]
        out_dict['input'] = pcds['input']
        out_dict['pred'] = pcds['input']
        out_dict['input_ref'] = pcds['input']
        
        out_dict['seg_mask'] = seg_mask 
        out_dict['seg_mask_ref'] = seg_mask 
        # out_dict['assigned_anchor'] = anchor_per_point.transpose(1, 2)
        out_dict['shift'] = pcds['shift']
        out_dict['scale'] = pcds['scale']
        return out_dict

    def get_params(self, pcds, device='cuda'):
        assert self.project_latent
        input = pcds['input'].to(device)
        attn_map = pcds['attn_map'].to(device) # ->[B, npoints, n_class]
        primary_part_latent = self.get_primary_latent(input, attn_map)
        _, shift, scale, _ = self.project_primary_latent(primary_part_latent)
        return shift, torch.log(scale)
    
    def get_primary_latent(self, input, attn=None):
        if self.include_attn_weight_in_encoder:
            assert attn is not None
            x = torch.cat([input, attn], dim=-1) # -> [B, npoints, n_class + 3]
        else:
            x = input


        primary_part_latent = self.encoder(x)
        return primary_part_latent

    def project_primary_latent(self, latent, interpolate=False, fix_shift=None, fix_scale=None, ref=None, mixing=False):
        B = latent.shape[0]
        if self.share_projection:
            if self.project_pe_type == 0:
                intrinsic, extrinsic = torch.split(self.down_proj(latent), [self.part_dim, self.param_dim], dim=-1)
            elif self.project_pe_type == 1:
                extrinsic = self.down_proj(latent)
                intrinsic = latent
            elif self.project_pe_type == 2:
                extrinsic = self.down_proj(latent.reshape(B, -1)).reshape(B, self.num_anchors, self.param_dim)
                intrinsic = latent
        else:
            intrinsic, extrinsic = torch.split(self.down_proj(latent.reshape(B, -1, 1)).reshape(B, self.num_anchors, self.part_dim + self.param_dim), [self.part_dim, self.param_dim], dim=-1)

        if self.learn_var:
            shift, scale = extrinsic.split(3, dim=-1)
        else:
            shift = extrinsic
            scale = torch.ones_like(shift)
        part_shifts = torch.zeros_like(shift)
        part_scales = torch.ones_like(scale)
        if self.training and self.global_shift:
            assert ref is not None
            rand_shift = torch.rand(B, 1, 3).cuda() - 0.5
            if self.vertical_only:
                rand_shift[..., [0,2]] = 0.
            part_shifts = rand_shift
            ref = ref + rand_shift
        if self.training and self.global_scale:
            assert ref is not None
            part_scales = torch.rand(B,1,3).cuda() / 2 + 0.7
            ref = ref * part_scales


        corrected_anchor = (shift + part_shifts) * part_scales 
        corrected_scale = scale + torch.log(part_scales)
        if interpolate:
            fix_shift = shift[0]
            fix_scale = scale[0]
        if mixing:
            fix_shift = torch.stack([shift[i, i] for i in range(self.num_anchors)], dim=0)
            fix_scale = torch.stack([scale[i, i] for i in range(self.num_anchors)], dim=0)
        if fix_shift is not None:
            while len(fix_shift.shape) < len(shift.shape):
                fix_shift = fix_shift.unsqueeze(0)
            fix_shift = fix_shift.expand([i if fix_shift.shape[id] == 1 else -1 for id, i in enumerate(shift.shape)])
            for i, j in zip(fix_shift.shape, shift.shape):
                assert i == j
            corrected_anchor = fix_shift
        if fix_scale is not None:
            while len(fix_scale.shape) < len(scale.shape):
                fix_scale = fix_scale.unsqueeze(0)
            fix_scale = fix_scale.expand([i if fix_scale.shape[id] == 1 else -1 for id, i in enumerate(scale.shape)])
            for i, j in zip(fix_scale.shape, scale.shape):
                assert i == j
            corrected_scale = fix_scale
        params = torch.cat([corrected_anchor, corrected_scale], dim=-1) if self.learn_var else corrected_anchor

        if self.share_projection:
            _primary_part_latent = intrinsic + self.up_proj(params)
        else:
            _primary_part_latent = intrinsic + self.up_proj(params.reshape(B, -1, 1)).reshape(B, self.num_anchors, self.part_dim)

        if self.post_ff:
            latent = latent + self.out_layers(_primary_part_latent.transpose(1,2)).transpose(1,2)
        else:
            latent = _primary_part_latent
        if self.use_log_for_scale:
            corrected_scale = torch.exp(corrected_scale)
        else:
            corrected_scale = F.relu(corrected_scale)
        return latent, corrected_anchor, torch.exp(corrected_scale), ref
    
    def gather_all(self, anchor_assignments, part_latent, anchors=None, variances=None):
        B = anchor_assignments.shape[0]
        latent_per_point = gather_operation(part_latent.transpose(2,1).contiguous(), anchor_assignments).reshape(B, -1, self.npoints).transpose(1,2) 
        if anchors is not None:
            anchor_per_point = gather_operation(anchors.transpose(2, 1).contiguous(), anchor_assignments).reshape(B, 3, self.npoints)  
        if variances is not None and  not isinstance(variances, float):
            variance_per_point = gather_operation(variances.transpose(2, 1).contiguous(), anchor_assignments).reshape(B, 3, self.npoints).transpose(1,2) 
        else:
            variance_per_point = 1.
        assert anchor_per_point.shape[1] == 3
        
        return latent_per_point, anchor_per_point, variance_per_point
    def modify_grad(self, epoch):
        if self.freeze_diffusion and (epoch % self.cache_interval >= self.freeze_interval):
            for param in self.diffusion.parameters():
                param.grad = None
            for param in self.decomposer.parameters():
                param.grad = None
        if self.freeze_encoder and (epoch % self.cache_interval < self.freeze_interval):
            for param in self.encoder.parameters():
                param.grad = None
            for param in self.down_proj.parameters():
                param.grad = None
            for param in self.up_proj.parameters():
                param.grad = None
    def forward(self, pcds, device='cuda', iter=-1, epoch=-1, fixing_shift=None, fixing_scale=None):
        """
            input: [B, npoints, 3]
            attn_map: [B, npoints, n_class]
            seg_mask: [B, npoints]
        """
        if self.save_weights:
            self.eval()
            print(1)
            torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, "encoder_ckpt.pth"))
            torch.save(self.decomposer.state_dict(), os.path.join(self.save_dir, "decomposer_ckpt.pth"))
            exit()
        if self.language_edit:
            return self.language_edit_step(pcds, device)
        input = pcds['input'].to(device)
        ref = pcds['ref'].to(device)
        attn_map = pcds['attn_map'].to(device) # ->[B, npoints, n_class]
        seg_mask = pcds['seg_mask'].to(device)
        loss_dict = dict()


        primary_part_latent = self.get_primary_latent(input, attn=attn_map)
        if self.project_latent:
            if self.intervaled_training and (epoch % self.cache_interval < self.freeze_interval):
                primary_part_latent, corrected_anchor, corrected_scale, _ = self.project_primary_latent(primary_part_latent, fix_shift=fixing_shift, fix_scale=fixing_scale)
            else:
                primary_part_latent, corrected_anchor, corrected_scale, ref = self.project_primary_latent(primary_part_latent, ref=ref)
                
        global_feature, anchors, secondary_part_latent = self.decomposer(primary_part_latent)
        assert anchors.shape[1] == secondary_part_latent.shape[1] == self.num_anchors
        variances = 1.
        if self.use_primary:
            part_latent = primary_part_latent 
        else:
            part_latent = secondary_part_latent
        if self.project_latent:
            anchors = corrected_anchor
            if self.learn_var:
                variances = corrected_scale
        if self.use_zero_anchor:
            anchors = torch.zeros_like(anchors)
        if self.use_global_anchor:
            anchors = pcds['global_anchor_mean'].cuda()
        if self.use_gt_anchors:
            anchors = pcds['part_means'].cuda()[..., 1:]
        
        if self.training:
            if self.guidance:
                idxs = torch.rand(part_latent.shape[:-1]).to(part_latent.device) >= self.part_latent_dropout_prob
                part_latent = part_latent * idxs.unsqueeze(-1)

                global_idxs = torch.rand(global_feature.shape[0]).to(part_latent.device) >= self.global_latent_dropout_prob
                global_feature = global_feature * global_idxs.unsqueeze(-1)

            t, _ = self.sampler.sample(input.shape[0], device)
            latent_per_point, anchor_per_point, variance_per_point = self.gather_all(seg_mask.to(torch.int32), part_latent, anchors, variances)
            
            if self.anchor_weight_annealing:
                anchor_loss_weight = self.ratio * min(epoch, self.annealing_epoch) + self.start_weight
            elif self.use_zero_anchor or self.use_global_anchor or self.use_gt_anchors:
                anchor_loss_weight = 0
            else:
                anchor_loss_weight = self.anchor_loss_weight
            anchor_loss = self.loss_func(ref, anchor_per_point.transpose(1,2), var = variance_per_point)
            loss_dict['anchor_loss'] = anchor_loss_weight * anchor_loss
            loss_dict['anchor_weight'] = torch.ones(1).cuda() * anchor_loss_weight
            
            if self.normal_diffusion:
                anchor_per_point = anchor_per_point * 0
                variance_per_point = torch.ones_like(variance_per_point)

            if self.detach_anchor:
                anchor_per_point = anchor_per_point.detach()
            if self.detach_variance and not isinstance(variance_per_point, float):
                variance_per_point = variance_per_point.detach()
            if self.freeze_diffusion and (epoch % self.cache_interval > self.freeze_interval):
                with torch.no_grad():
                    diffusion_loss = self.diffusion.training_losses(ref.transpose(1,2), t, anchors=anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, variance=variance_per_point, reduce=True, anchor_assignment=seg_mask.to(torch.int32))
            else:
                diffusion_loss = self.diffusion.training_losses(ref.transpose(1,2), t, anchors=anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, variance=variance_per_point, reduce=True, anchor_assignment=seg_mask.to(torch.int32))
                
            loss_dict.update(diffusion_loss)
            return loss_dict

        else:
            
            if self.interpolate and not self.training:
                return self.interpolate_latent(seg_mask, device, pcds)
            if self.combine and not self.training:
                ids = [15, 50, 14,19]
                return self.combine_latent(x, pcds, seg_mask, ids, device)
            if self.drift_anchors:
                return self.anchor_drift(pcds, x, global_feature, anchors, part_latent, seg_mask, device)

            if self.sample_by_seg_mask:
                anchor_assignments = seg_mask.to(torch.int32)
                multiple = int(self.npoints / anchor_assignments.shape[1])
                anchor_assignments = anchor_assignments.repeat_interleave(multiple, dim=1)
                latent_per_point, anchor_per_point, variance_per_point = self.gather_all(anchor_assignments, part_latent, anchors, variances)

            else:
                B = input.shape[0]
                anchor_per_point = anchors.reshape(B, self.num_anchors, 1, -1).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, -1).transpose(1,2)
                assert anchor_per_point.shape[1] == 3
                latent_per_point = part_latent.reshape(B, self.num_anchors, 1, 3).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, 3)
                if not isinstance(variances, float):
                    variance_per_point = variances.reshape(B, self.num_anchors, 1, 3).repeat([1, 1, self.points_per_anchor, 1]).reshape(B, self.npoints, 3).transpose(1,2)
                else:
                    variance_per_point = 1.
                anchor_assignments=torch.arange(self.num_anchors).reshape(1, -1, 1).expand(B, -1, self.points_per_anchor).reshape(B, self.npoints)
            if self.normal_diffusion:
                anchor_per_point = anchor_per_point * 0
                variance_per_point = torch.ones_like(variance_per_point)

           
            pred = self.q_sample(ref.transpose(2,1), anchors=anchor_per_point, device=device, variance=None) if self.forward_sample else self.decode(anchor_per_point, code=global_feature, pointwise_latent=latent_per_point, device=device, variance=variance_per_point, anchor_assignments=anchor_assignments) 
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
