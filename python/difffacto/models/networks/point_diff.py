import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Module, functional
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS, DIFFUSIONS, SAMPLERS, METRICS, DECOMPOSERS, SEGMENTORS
from pointnet2_ops.pointnet2_utils import gather_operation
from .language_utils.language_util import tokenizing
VOCAB_SIZE=2787

@MODELS.register_module()
class PDM(Module):

    def __init__(
        self, 
        encoder, 
        diffusion, 
        sampler, 
        num_timesteps, 
        npoints=2048, 
        include_attn_weight_in_encoder=True,

        ret_traj=False, 
        ret_interval=20, 
        interpolate=False,
        combine=False,
        drift_anchors=False,
        save_pred_xstart=False,
        partglot_dataset=False,
        save_dir=None,
        save_weights=False,
        ):
        super().__init__()
        self.encoder = build_from_cfg(encoder, ENCODERS)
        self.diffusion = build_from_cfg(diffusion, DIFFUSIONS, num_timesteps=num_timesteps)
        self.sampler = build_from_cfg(sampler, SAMPLERS, num_timesteps=num_timesteps)
        self.save_weights=save_weights
        self.save_dir=save_dir
        self.drift_anchors=drift_anchors
        self.include_attn_weight_in_encoder=include_attn_weight_in_encoder
        self.num_timesteps = int(num_timesteps)     
        self.partglot_dataset=partglot_dataset
        
        self.npoints = npoints
        self.interpolate=interpolate
        self.ret_traj=ret_traj
        self.ret_interval=ret_interval
        self.save_pred_xstart=save_pred_xstart
        self.combine=combine

        print(self)

    
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

    def interpolate_latent(self, x, seg_mask, device, pcds):
        ids = [(19,20, 2), (17, 15, 0), (37, 38, 1), (122, 120, 1), (19, 18, 2), (66,65, 2)]
        out_dict = {f"interpolate sample {i}":[] for i in range(10)}
        # out_dict['combined'] = []
        # out_dict['seg_mask_ref'] = []
        out_dict['input1_ref'] = []
        out_dict['input2_ref'] = []
        out_dict['seg_mask1'] = []
        out_dict['seg_mask2'] = []
        out_dict.update({f"interpolate anchor sample {i}": [] for i in range(10)})
        
        for id1, id2, anchor_id in ids:
            gt1, gt2 = x[id1], x[id2]
            latent1 = self.encoder(gt1.unsqueeze(0))
            latent2 = self.encoder(gt2.unsqueeze(0))
            latent1, latent2 = latent1.squeeze(0), latent2.squeeze(0) # [1, n_class, C]

            dx = torch.zeros([10, self.num_anchors, 1]).to(device)
            dx[:, anchor_id] += torch.linspace(0, 1, steps=10).reshape(-1, 1).to(device)
            primary_latent = latent1.unsqueeze(0) + (latent2 - latent1).unsqueeze(0) * dx
            B = primary_latent.shape[0]
            if self.project_latent:
                if self.share_projection:
                    intrinsic, extrinsic = torch.split(self.down_proj(primary_latent), [self.part_dim, self.param_dim], dim=-1)
                else:
                    intrinsic, extrinsic = torch.split(self.down_proj(primary_latent.reshape(B, -1, 1)).reshape(B, self.num_anchors, self.part_dim + self.param_dim), [self.part_dim, self.param_dim], dim=-1)
                
                if self.learn_var:
                    shift, scale = extrinsic.split(3, dim=-1)
                    scale = scale[0].unsqueeze(0).expand(B, -1, -1)
                else:
                    shift = extrinsic    
                
                shift = shift[0].unsqueeze(0).expand(B, -1, -1)
                
                if self.learn_var:
                    if self.share_projection:
                        primary_latent = intrinsic + self.up_proj(torch.cat([shift, scale], dim=-1))
                    else:
                        primary_latent = intrinsic + self.up_proj(torch.cat([shift, scale], dim=-1).reshape(B, -1, 1)).reshape(B, self.num_anchors, self.part_dim)
            
            global_feat, new_anchors, secondary_latent = self.decomposer(primary_latent)
            if self.project_latent:
                new_anchors = shift 
                if self.learn_var:
                    new_vars = scale
            if self.use_primary:
                new_latents = primary_latent 
            else:
                new_latents = secondary_latent
            latent_per_point = new_latents.reshape(10, self.num_anchors, 1, -1).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, -1)
            anchor_per_point = new_anchors.reshape(10, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, 3).transpose(1,2)
            if self.learn_var:
                variance_per_point = new_vars.reshape(10, self.num_anchors, 1, 3).expand(-1, -1, self.points_per_anchor, -1).reshape(10, self.npoints, 3)
                variance_per_point = torch.exp(variance_per_point)
            else:
                variance_per_point=None

            pred = self.decode(anchor_per_point, code=global_feat, pointwise_latent=latent_per_point, device=device, variance=variance_per_point) 
            # ref = torch.zeros([self.npoints, 3]).cuda()
            # _ref = torch.cat([gt1[seg_mask[id1] != anchor_id, :3], gt2[seg_mask[id2] == anchor_id, :3]], dim=0)
            # ref[:_ref.shape[0]] = _ref
            # seg_mask_ref = torch.zeros_like(seg_mask[0])
            # _seg_mask_ref = torch.cat([seg_mask[id1][seg_mask[id1] != anchor_id], seg_mask[id2][seg_mask[id2] == anchor_id]], dim=0)
            # seg_mask_ref[:_seg_mask_ref.shape[0]] = _seg_mask_ref
            for i in range(pred['pred'].shape[0]):
                out_dict[f"interpolate sample {i}"].append(pred['pred'][i])
                out_dict[f"interpolate anchor sample {i}"].append(new_anchors[i])
            # out_dict['combined'].append(ref)
            # out_dict['seg_mask_ref'].append(seg_mask_ref)
            out_dict['input1_ref'].append(gt1[:, 0:3])
            out_dict['input2_ref'].append(gt2[:, 0:3])
            out_dict['seg_mask1'].append(seg_mask[id1])
            out_dict['seg_mask2'].append(seg_mask[id2])
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
        gts = []
        for i in ids:
            gts.append(x[i])
        gts = torch.stack(gts, dim=0)
        latents = self.encoder(gts)
        latents = torch.stack([latents[i, i] for i in range(self.num_anchors)], dim=0).reshape(1, self.num_anchors, -1)
        global_feature, new_params, new_latents = self.decomposer(latents)
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
        
    def forward(self, pcds, device='cuda', iter=-1):
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

        z = self.encoder(x).reshape(B, -1)
                
        if self.training:
            
            t, _ = self.sampler.sample(B, device)



            diffusion_loss = self.diffusion.training_losses(ref.transpose(1,2), t, anchors=torch.zeros_like(ref.transpose(1,2)), code=z, pointwise_latent=None, variance=None, reduce=True)
            loss_dict['diffusion_loss'] = diffusion_loss
            return loss_dict

        else:
            
            pred = self.decode(torch.zeros_like(ref.transpose(1,2)), code=z, pointwise_latent=None, device=device, variance=None) 
            pred['input'] = input
            pred['input_ref'] = ref
            pred['seg_mask'] = seg_mask 
            pred['seg_mask_ref'] = seg_mask 
            pred['shift'] = pcds['shift']
            pred['scale'] = pcds['scale']
            pred = {k:v.detach().cpu() for k, v in pred.items()}
            return pred
