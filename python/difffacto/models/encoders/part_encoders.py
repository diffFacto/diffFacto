import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from functools import partial
from difffacto.utils.registry import ENCODERS, DECOMPOSERS, build_from_cfg
from difffacto.models.encoders.pointnet import PointNet, PointNetVAEBase
from pointnet2_ops.pointnet2_utils import gather_operation
from einops import rearrange, repeat
from difffacto.utils.misc import gaussian_log_likelihood, reparameterize_gaussian, gaussian_entropy, standard_normal_logprob
from difffacto.models.diffusions.nets.attention import FeedForward, BasicTransformerBlock, zero_module, MLP
from .flow import build_latent_flow
def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

@ENCODERS.register_module()
class PartAlignerTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head, out_channels,
                 depth=1, dropout=0., use_linear=False, n_class=4,
                 use_checkpoint=False, single_attn=False, class_cond=True,
                 mask_out_unreferenced_code=True, cimle=False, noise_dim=32, noise_scale=10, cimle_start_epoch=0,
                 add_class_cond=False, cond_noise_type=0, cond_noise_as_token=False, 
                ):
        super().__init__()
        self.n_class=n_class
        self.cimle=cimle
        self.add_class_cond=add_class_cond
        self.noise_scale=noise_scale
        self.cond_noise_type=cond_noise_type 
        if cond_noise_as_token:
            self.cond_noise_type = 1
            print("cond noise as token Deprecated!! use cond_noise_type=1 instead")
        in_channels = in_channels + int(class_cond and not add_class_cond) * n_class + int(cimle and self.cond_noise_type in [0, 4]) * noise_dim
        self.in_channels = in_channels
        self.class_cond=class_cond
        self.mask_out_unreferenced_code=mask_out_unreferenced_code
        inner_dim = n_heads * d_head
        self.noise_dim=noise_dim if self.cond_noise_type!=1 else inner_dim
        self.cimle_start_epoch=cimle_start_epoch
        self.inner_dim = inner_dim
        if class_cond and add_class_cond:
            self.class_emb = nn.Embedding(n_class, inner_dim)
        self.pre_norm = nn.LayerNorm(inner_dim)
        self.post_norm = nn.LayerNorm(inner_dim)
        if self.cond_noise_type == 3:
            self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    self.noise_dim,
                    2 * inner_dim,
                ),
            )
        if self.cond_noise_type == 4:
            self.emb_layer = MLP([self.noise_dim, 4*self.noise_dim,4*self.noise_dim, self.noise_dim], use_linear=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=None,
                                   checkpoint=use_checkpoint, single_attn=single_attn, adaln=(self.cond_noise_type==2), y_dim=self.noise_dim)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = nn.Conv1d(inner_dim, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_out = nn.Linear(inner_dim, out_channels)
        self.use_linear = use_linear

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, noise: torch.Tensor=None):
        assert x.shape[-1] == self.n_class
        if self.class_cond and not self.add_class_cond:
            class_embed = torch.eye(self.n_class).to(x).unsqueeze(0).repeat_interleave(x.shape[0], dim=0) # B, n_ctx, n_ctx
            x = torch.cat([x, class_embed], dim=1)
        if self.cimle and self.cond_noise_type != 1:
            assert noise is not None
            if noise.shape[1] != self.noise_dim:
                noise = torch.zeros(x.shape[0], self.noise_dim).to(x)
            noise = noise * self.noise_scale
            if self.cond_noise_type == 4:
                noise = self.emb_layer(noise)
            if self.cond_noise_type in [0, 4]:
                x = torch.cat([x, noise.unsqueeze(-1).expand(-1, -1, self.n_class)], dim=1)
        assert self.in_channels == x.shape[1]
        h =  self._forward_attn(x, mask=mask if self.mask_out_unreferenced_code else None, noise=noise if self.cimle else None)
        mean, logvar = torch.split(h, 3, dim=1)
        return mean, logvar
    
    def _forward_attn(self, x, mask=None, noise=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c n -> b n c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        if self.class_cond and self.add_class_cond:
            class_emb = self.class_emb(torch.arange(self.n_class, device=x.device)).unsqueeze(0)
            x = x + class_emb
        if self.cimle:
            if self.cond_noise_type==1:
                new_mask = torch.zeros(x.shape[0], self.n_class + 1).to(mask)
                new_mask[:, 1:] = mask 
                if noise.shape[1] != self.noise_dim:
                    noise = torch.randn(x.shape[0], self.noise_dim).to(x)
                else:
                    new_mask[:, 0] = 1.0
                noise = noise * self.noise_scale
                x = torch.cat([noise.unsqueeze(1), x], dim=1)
                mask = new_mask
                x = self.pre_norm(x)
            elif self.cond_noise_type == 3:
                shift, scale = torch.chunk(self.emb_layer(noise), 2, dim=-1)
                x = self.pre_norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            x = self.pre_norm(x)    
        
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=None, mask=mask, y=noise if self.cond_noise_type==2 else None)
        x = self.post_norm(x)
        if self.cimle and self.cond_noise_type==1:
            x = x[:, 1:]
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b n c -> b c n').contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return  x


class AdaInstanceNorm1d(nn.Module):
    def __init__(self, dim, ctx_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim)
        self.proj = nn.Sequential(
            nn.Linear(ctx_dim, dim * 4), 
            nn.SiLU(),
            nn.Linear(dim * 4, 2* dim)
        )
        self.dim=dim
        with torch.no_grad():
            self.proj[0].weight.zero_()
            self.proj[0].bias.zero_()
            self.proj[2].weight.zero_()
            self.proj[2].bias.zero_()
    def forward(self, x, ctx):
        x = self.norm(x)
        shift, scale = torch.split(self.proj(ctx), self.dim, dim=1)
        return x * (1 + scale) + shift

@ENCODERS.register_module()
class PartAlignerCIMLE(nn.Module):
    def __init__(
        self,
        width: int,
        n_class: int,
        param_dim:int=3,
        noise_dim:int=32,
        norm: str='bn',
        noise_encoder_type: int=0
    ):
        super().__init__()
        self.width=width 
        self.n_class = n_class
        self.noise_dim=noise_dim
        self.norm = norm
        if norm == 'bn':
            enc_norm = nn.BatchNorm1d 
            norm_layer = nn.BatchNorm1d 
        elif norm == 'in':
            enc_norm = nn.InstanceNorm1d 
            norm_layer = nn.InstanceNorm1d 
        elif norm == 'adain':
            enc_norm = nn.InstanceNorm1d 
            norm_layer = partial(AdaInstanceNorm1d, ctx_dim=noise_dim)
        self.noise_encoder_type=noise_encoder_type
        if noise_encoder_type == 0:
            self.noise_encoder=nn.Sequential(
                nn.Linear(width*n_class + noise_dim, 512),
                enc_norm(512),
                nn.ReLU(False),
                nn.Linear(512, 128),
                enc_norm(128),
                nn.ReLU(False),
                nn.Linear(128, noise_dim),
            )
        self.fc1_m = nn.Linear(width*n_class+ noise_dim if norm != 'adain' else width*n_class, 512)
        self.fc2_m = nn.Linear(512, 256)
        self.fc3_m = nn.Linear(256, 64)
        self.fc4_m = nn.Linear(64, param_dim*n_class)
        self.fc_bn1_m = norm_layer(512)
        self.fc_bn2_m = norm_layer(256)
        self.fc_bn3_m = norm_layer(64)
        
        
        self.fc1_v = nn.Linear(width*n_class+ noise_dim if norm != 'adain' else width*n_class, 512)
        self.fc2_v = nn.Linear(512, 256)
        self.fc3_v = nn.Linear(256, 64)
        self.fc4_v = nn.Linear(64, param_dim*n_class)
        self.fc_bn1_v = norm_layer(512)
        self.fc_bn2_v = norm_layer(256)
        self.fc_bn3_v = norm_layer(64)
        for i in range(1, 5):
            for k in ['m', 'v']:
                init_linear(getattr(self, f"fc{i}_{k}"), 0.25)
    
    
    
    def forward(self, part_code, valid_id, noise):
        if noise.shape[1] != self.noise_dim:
            noise = torch.randn(part_code.shape[0], self.noise_dim).to(part_code.device)
        # print(noise.shape)
        assert noise.shape[1] == self.noise_dim
        assert part_code.shape[1:] == (self.width, self.n_class)
        part_code = part_code * valid_id.unsqueeze(1)
        B = part_code.shape[0]
        part_code = part_code.reshape(B, -1)
        if self.noise_encoder_type == 0:
            noise = self.noise_encoder(torch.cat([part_code, noise], dim=-1))
        if self.norm in ['bn', 'in']:
            noised_part_code = torch.cat([part_code, noise], dim=-1)
            m = F.relu(self.fc_bn1_m(self.fc1_m(noised_part_code)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = F.relu(self.fc_bn3_m(self.fc3_m(m)))
            m = self.fc4_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(noised_part_code)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = F.relu(self.fc_bn3_v(self.fc3_v(v)))
            v = self.fc4_v(v)
        elif self.norm in ['adain']:
            m = F.relu(self.fc_bn1_m(self.fc1_m(part_code), noise))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m), noise))
            m = F.relu(self.fc_bn3_m(self.fc3_m(m), noise))
            m = self.fc4_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(part_code), noise))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v), noise))
            v = F.relu(self.fc_bn3_v(self.fc3_v(v), noise))
            v = self.fc4_v(v)
        return m.reshape(B, 3, self.n_class), v.reshape(B, 3, self.n_class)

@ENCODERS.register_module()
class PartAligner(nn.Module):
    def __init__(
        self, 
        n_class: int, 
        width: int):
        super().__init__()

        self.width=width 
        self.n_class = n_class
        self.fc1_m = nn.Linear(width*n_class, 512)
        self.fc2_m = nn.Linear(512, 256)
        self.fc3_m = nn.Linear(256, 64)
        self.fc4_m = nn.Linear(64, 3*n_class)
        self.fc_bn1_m = nn.BatchNorm1d(512)
        self.fc_bn2_m = nn.BatchNorm1d(256)
        self.fc_bn3_m = nn.BatchNorm1d(64)
        
        
        self.fc1_v = nn.Linear(width*n_class, 512)
        self.fc2_v = nn.Linear(512, 256)
        self.fc3_v = nn.Linear(256, 64)
        self.fc4_v = nn.Linear(64, 3*n_class)
        self.fc_bn1_v = nn.BatchNorm1d(512)
        self.fc_bn2_v = nn.BatchNorm1d(256)
        self.fc_bn3_v = nn.BatchNorm1d(64)
        for i in range(1, 5):
            for k in ['m', 'v']:
                init_linear(getattr(self, f"fc{i}_{k}"), 0.25)
        
    def forward(self, part_code, valid_id=None, noise=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        assert part_code.shape[1:] == (self.width, self.n_class)
        part_code = part_code * valid_id.unsqueeze(1)
        B = part_code.shape[0]
        part_code = part_code.reshape(B, -1)
        m = F.relu(self.fc_bn1_m(self.fc1_m(part_code)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = F.relu(self.fc_bn3_m(self.fc3_m(m)))
        m = self.fc4_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(part_code)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = F.relu(self.fc_bn3_v(self.fc3_v(v)))
        v = self.fc4_v(v)
        return m.reshape(B, 3, self.n_class), v.reshape(B, 3, self.n_class)

@ENCODERS.register_module()
class PartEncoder(nn.Module):
    def __init__(
        self, 
        encoder=None, 
        n_class=4, 
        part_aligner=None, 
        fit_loss_weight=1.0, 
        include_z=True,
        include_part_code=False,
        include_params=False,
        use_gt_params=False,
        encode_ref=False,
        scale_var=1.0,
        fit_loss_type=0,
        origin_scale=False,
        kl_weight=0.001, 
        use_flow=False,
        latent_flow_depth=14, 
        latent_flow_hidden_dim=256, 
        use_gt_params_in_training=False,
        gen=False,
        gt_param_annealing=False,
        gt_param_annealing_start_epoch=500,
        gt_param_annealing_end_epoch=1000,
        kl_weight_annealing=False,
        min_kl_weight=1e-7,
        include_class_label=False,
        kl_weight_annealing_end_epoch=3000,
        normalize_part_code=False,
        detach_params_in_ctx=False,
        prior_var=1.0,
        per_part_encoder=False,
        selective_noise_sampling=False,
        selective_noise_sampling_global=False,
        ):
        super().__init__()
        self.per_part_encoder=per_part_encoder
        if not per_part_encoder:
            self.encoder = build_from_cfg(encoder, ENCODERS, num_anchors=n_class)
            self.zdim = self.encoder.zdim
        else:
            self.encoder = nn.ModuleList([build_from_cfg(encoder, ENCODERS, num_anchors=1) for _ in range(n_class)])
            self.zdim = self.encoder[0].zdim
        self.part_aligner = build_from_cfg(part_aligner, ENCODERS)
        self.include_z=include_z
        self.include_part_code=include_part_code
        self.include_params=include_params
        self.detach_params_in_ctx=detach_params_in_ctx
        self.n_class = n_class
        self.encode_ref=encode_ref
        self.prior_var=prior_var
        if encode_ref:
            self.ref_encoder = PointNet(zdim=self.encoder.zdim, point_dim=3, num_anchors=1)
        self.log_scale_var=math.log(scale_var)
        self.fit_loss_weight=fit_loss_weight
        self.use_gt_params=use_gt_params
        self.fit_loss_type = fit_loss_type
        self.selective_noise_sampling=selective_noise_sampling
        self.selective_noise_sampling_global=selective_noise_sampling_global
        self.origin_scale=origin_scale
        self.gen=gen 
        self.include_class_label=include_class_label
        self.normalize_part_code=normalize_part_code
        self.gt_param_annealing=gt_param_annealing
        self.gt_param_annealing_start_epoch=gt_param_annealing_start_epoch
        self.gt_param_annealing_end_epoch=gt_param_annealing_end_epoch
        self.use_gt_params_in_training=use_gt_params_in_training
        if gen:
            self.kl_weight=kl_weight
            self.kl_weight_annealing=kl_weight_annealing
            self.min_kl_weight=min_kl_weight
            self.kl_weight_annealing_end_epoch=kl_weight_annealing_end_epoch
            self.use_flow=use_flow
            if use_flow:
                self.flow = nn.ModuleList([build_latent_flow(latent_flow_depth, latent_flow_hidden_dim, self.zdim) for _ in range(self.n_class)])
    
    def sample_noise(self, pcds, device, num):
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        B, N, C = input.shape
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        if self.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)  # (B, F)
        else:
            part_code = part_code_means.transpose(1,2)
        noise = torch.randn(B*num, self.part_aligner.noise_dim).to(device)
        part_code, valid_id, seg_mask, ref, gt_shift, gt_var = map(lambda t: t.repeat_interleave(num, dim=0), [part_code, valid_id, seg_mask, ref, gt_shift, gt_var])
        mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise=noise, gt_mean=gt_shift, gt_var=gt_var, ref=ref)
        assert part_code.shape[-1] == self.n_class
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        # print(part_code.mean(), mean.mean(), logvar.mean(), fit_loss.mean(), gt_shift.mean(), gt_var.mean())
        id = fit_loss.reshape(B, num).min(1)[1]
        return noise.reshape(B, num, -1), id
    
    def gather_all(self, anchor_assignments, anchors=None, variances=None, valid_id=None):
        B, N = anchor_assignments.shape
        anchor_per_point = torch.zeros(B, 3, N).to(anchor_assignments.device) 
        variance_per_point = torch.zeros(B, 3, N).to(anchor_assignments.device) 
        flag_per_point = torch.ones(B, 1, N).to(anchor_assignments.device) 
        if anchors is not None:
            anchor_per_point = gather_operation(anchors.contiguous(), anchor_assignments).reshape(B, 3, N)  
        if variances is not None:
            variance_per_point = gather_operation(variances.contiguous(), anchor_assignments).reshape(B, 3, N) 
        if valid_id is not None:
            flag_per_point = gather_operation(valid_id.unsqueeze(1).contiguous(), anchor_assignments).reshape(B, 1, N) 
        return anchor_per_point, variance_per_point, flag_per_point
    def get_part_code(self, input, seg_flag):
        B = input.shape[0]
        if self.per_part_encoder:
            m, v = [], []
            for i in range(self.n_class):
                idx = seg_flag[..., i] != 1
                if not torch.all(idx):
                    _m, _v = self.encoder[i](input, idx)
                    m.append(_m)
                    v.append(_v)
                else:
                    m.append(torch.zeros(B, self.zdim, 1).cuda())
                    v.append(torch.zeros(B, self.zdim, 1).cuda())
            m = torch.stack(m, dim=1)
            v = torch.stack(v, dim=1)
            return m, v
        return self.encoder(input, seg_flag)
            
    def get_params_from_part_code(self, part_code, valid_id, gt_mean=None, gt_var=None, ref=None, noise=None, **kwargs):
        if self.part_aligner is not None:
            if self.encode_ref:
                global_feat = self.ref_encoder(ref.transpose(1,2)).expand(-1, self.n_class, -1).transpose(1,2)
                mean, logvar = self.part_aligner(global_feat, valid_id, noise=noise, **kwargs)
            else:
                mean, logvar = self.part_aligner(part_code, valid_id, noise=noise, **kwargs)
        else:
            mean = logvar = None
        if self.use_gt_params:
            logvar = torch.log(gt_var) # this is std, we need convert it to var
            mean = gt_mean
        return  mean, logvar


    def prepare_ctx(self, part_code, mean, logvar, anchor_assignments=None, z=None):
        ctx = []
        B = part_code.shape[0]
        npoint = anchor_assignments.shape[1]
            
        if self.include_z:
            ctx.append(part_code.reshape(B, -1).unsqueeze(1).repeat_interleave(npoint, dim=1))
        if self.include_part_code:
            latent_per_point = gather_operation(part_code.contiguous(), anchor_assignments).reshape(B, part_code.shape[1], npoint)
            ctx.append(latent_per_point.transpose(1,2))
        if self.include_class_label:
            seg_flag = gather_operation(torch.eye(self.n_class).cuda().unsqueeze(0).repeat_interleave(B, dim=0).contiguous(), anchor_assignments).reshape(B, self.n_class, npoint)
            ctx.append(seg_flag.transpose(1,2))
        if self.include_params:                    
            if mean is not None:
                mean_per_point = gather_operation(mean.contiguous(), anchor_assignments).reshape(B, 3, npoint)
                if self.detach_params_in_ctx:
                    mean_per_point = mean_per_point.detach()
                ctx.append(mean_per_point.transpose(1,2))
                
            if logvar is not None:
                logvar_per_point = gather_operation(logvar.contiguous(), anchor_assignments).reshape(B, 3, npoint)
                if self.detach_params_in_ctx:
                    logvar_per_point = logvar_per_point.detach()
                ctx.append(torch.exp(logvar_per_point + self.log_scale_var).transpose(1,2))
        return ctx
    
    def get_fit_loss(self, ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask):
        if self.part_aligner is not None:
            if self.fit_loss_type == 0:
                fit_loss = -gaussian_log_likelihood(ref, mean_per_point, logvar_per_point).mean(1, keepdim=True) * flag_per_point
                fit_loss = fit_loss.sum(dim=(-1, -2)) / flag_per_point.sum(dim=(-1, -2))
            elif self.fit_loss_type == 1:
                gt_mean = gt_shift
                gt_var = gt_var
                fit_loss = F.mse_loss(torch.cat([mean, torch.exp(logvar)], dim=1), torch.cat([gt_mean, gt_var], dim=1), reduction='none')
                fit_loss = fit_loss * valid_id.unsqueeze(1)
                fit_loss = fit_loss.sum(dim=(-1, -2)) / valid_id.sum(dim=-1)
            elif self.fit_loss_type == 2:
                gt_logvar = torch.log(gt_var)
                prior_samples = reparameterize_gaussian(mean, logvar)
                p_z = -gaussian_log_likelihood(prior_samples, gt_shift, gt_logvar, dim=3).mean(1)
                entropy = -gaussian_entropy(logvar, dim=1)
                fit_loss = p_z + entropy
                fit_loss = fit_loss * valid_id
                fit_loss = fit_loss.sum(dim=1) / valid_id.sum(dim=1)
            elif self.fit_loss_type == 3:
                gt_mean_per_point = gather_operation(gt_shift, seg_mask)
                gt_var_per_point = gather_operation(gt_var, seg_mask)
                samples = torch.randn_like(gt_var_per_point) * torch.sqrt(gt_var_per_point) + gt_mean_per_point
                fit_loss = -gaussian_log_likelihood(samples, mean_per_point, logvar_per_point).mean(1, keepdim=True) * flag_per_point
                fit_loss = fit_loss.sum(dim=(-1, -2)) / flag_per_point.sum(dim=(-1, -2))
            elif self.fit_loss_type == 4:
                gt_mean = gt_shift
                gt_logvar = torch.log(gt_var)
                fit_loss = F.mse_loss(torch.cat([mean, logvar], dim=1), torch.cat([gt_mean, gt_logvar], dim=1), reduction='none')
                fit_loss = fit_loss * valid_id.unsqueeze(1)
                fit_loss = fit_loss.sum(dim=(-1, -2)) / valid_id.sum(dim=-1)
        else:
            fit_loss = torch.zeros([1]).to(ref)
        return fit_loss
    
    # def sample_latents(self, sample_num, sample_points, device, fixed_id=None, valid_id=None, **kwargs):
    #     part_code = torch.randn(sample_num, self.zdim, self.n_class).to(device)
    #     fixed_codes = part_code[0].unsqueeze(0)
    #     part_code = part_code * (1 - fixed_id).unsqueeze(0).unsqueeze(0) + fixed_id.unsqueeze(0).unsqueeze(0) * fixed_codes
    #     if self.use_flow:
    #         part_code_ = []
    #         for i in range(self.n_class):
    #             _part_code = part_code[..., i]
    #             part_code_.append(self.flow[i](_part_code, reverse=True).view(sample_num, self.zdim))
    #         part_code = torch.stack(part_code_, dim=-1)
    #     if valid_id is None:
    #         valid_id = torch.ones(sample_num, self.n_class).to(part_code)

    #     mean, logvar = self.get_params_from_part_code(part_code, valid_id, **kwargs)
        
    #     assert part_code.shape[-1] == self.n_class
    #     ids = torch.arange(self.n_class).to(device).unsqueeze(0) * valid_id
    #     seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(sample_num, sample_points)
    #     mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=mean, variances=logvar + self.log_scale_var)
    #     ctx = self.prepare_ctx(part_code, mean, logvar + self.log_scale_var, anchor_assignments=seg_mask)
    #     return ctx, mean_per_point, logvar_per_point, seg_mask, valid_id
    def subsample_params(self, mean, logvar, valid_id, num, return_ids=False):
        print('Selective Down Sampling----------')
        B = mean.shape[0]
        K = mean.shape[1]
        param = torch.cat([mean, logvar], dim=2)
        param_score = torch.cat([mean, logvar], dim=2).reshape(B*K, 6, 4)
        outs = []
        idss = []
        expand_valid_id = valid_id.repeat_interleave(K, dim=0)
        for i in range(B*K):
            points = torch.randn(512, 3, self.n_class).cuda() * torch.exp(0.5 * logvar.reshape(B*K, 3, 4)[i][None]) + mean.reshape(B*K, 3, 4)[i][None]
            _points = points[..., expand_valid_id[i].bool()].transpose(1,2).reshape(-1, 3)
            shift = (_points.max(0)[0] + _points.min(0)[0]) / 2
            scale = (_points.max(0)[0] - _points.min(0)[0]).max() / 2
            points = (points - shift.reshape(1, 3, 1)) / scale.reshape(1, 1, 1)
            param_score[i] = torch.cat([points.mean(0), 2.0 * torch.log(points.std(0))], dim=0)
        param_score = param_score.reshape(B, K, 6, 4)
        for k in range(B):
            _param = param[k]
            _param_score = param_score[k]
            _valid_id = valid_id[k]
            out = _param[0][None]
            out_score = _param_score[0][None]
            selected = [0]
            while out.shape[0] < num:
                dists = []
                for i in range(_param.shape[0]):
                    if i in selected: continue
                    p = _param_score[i]
                    d = F.mse_loss(p[None].expand(out.shape[0], -1, -1), out_score, reduction='none')
                    d = d * _valid_id.unsqueeze(0).unsqueeze(0)
                    d = d.sum(dim=(-1, -2)) / _valid_id.sum(dim=-1)
                    dists.append((i, d.min()))
                ids, ds = zip(*dists)
                selected_id = ids[torch.argmax(torch.tensor(ds))]
                selected.append(selected_id)
                out = torch.cat([out, _param[selected_id][None]], dim=0)
                out_score = torch.cat([out_score, _param_score[selected_id][None]], dim=0)
                # print(out.shape)
            outs.append(out)
            idss.append(torch.tensor(selected).cuda())
        print(len(outs), len(idss))
        if return_ids:
            return torch.chunk(torch.stack(outs, dim=0).reshape(-1, 6, self.n_class), 2, dim=1), idss
        return torch.chunk(torch.stack(outs, dim=0).reshape(-1, 6, self.n_class), 2, dim=1)

    def subsample_params_global(self, mean, logvar, valid_id, num):
        B = mean.shape[0]
        param = torch.cat([mean, logvar], dim=1)
        param_score = torch.cat([mean, logvar], dim=1)
        for i in range(B):
            points = torch.randn(512, 3, self.n_class).cuda() * torch.exp(0.5 * logvar[i][None]) + mean[i][None]
            _points = points[..., valid_id[i].bool()].transpose(1,2).reshape(-1, 3)
            shift = (_points.max(0)[0] + _points.min(0)[0]) / 2
            scale = (_points.max(0)[0] - _points.min(0)[0]).max() / 2
            points = (points - shift.reshape(1, 3, 1)) / scale.reshape(1, 1, 1)
            param_score[i] = torch.cat([points.mean(0), 2.0 * torch.log(points.std(0))], dim=0)
        out = param[0][None]
        out_score = param_score[0][None]
        selected_ids = [0]
        selected = torch.ones(B).cuda()
        selected_valid = valid_id[0][None]
        selected[0] = 0
        while out.shape[0] < num:
            d = F.mse_loss(param_score.unsqueeze(1).expand(-1, out_score.shape[0], -1, -1), out_score[None].expand(B, -1, -1, -1), reduction='none')
            # d[:, :, :3] = 0
            d = d * valid_id.unsqueeze(1).unsqueeze(1) * selected_valid.unsqueeze(1).unsqueeze(0)
            d = d.sum(dim=(-1, -2)) / (valid_id.unsqueeze(1) * selected_valid.unsqueeze(0)).sum(-1)
            d = d.min(1)[0]
            d = d * selected.reshape(B)
            selected_id = torch.argmax(d)
            selected[selected_id] = 0
            out = torch.cat([out, param[selected_id][None]], dim=0)
            selected_ids.append(selected_id)
            selected_valid = torch.cat([selected_valid, valid_id[selected_id][None]], dim=0)
            # print(out.shape)
        return torch.chunk(out, 2, dim=1), torch.tensor(selected_ids).long()
    
    def sample_with_fixed_latents(
        self, 
        codes, 
        valid_id, 
        gt_mean, 
        gt_logvar, 
        seg_mask, 
        sample_part_id, 
        how_many_each, 
        fix_size, 
        param_sample_num, 
        selective_param_sample):
        # import pickle
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp655_5e-4_debugged_adamax_n50/val/sample_airplane_n50_encode_1000.pkl", "rb"))
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_debugged_lamp_1e-3_n10_/val/sample_lamp_n10_4000.pkl", "rb"))
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/sample_chairs_from_lion_1250.pkl", "rb"))
        # codes = torch.from_numpy(data['part_latents']).cuda()
        # valid_id = torch.from_numpy(data['valid_id']).cuda()
        # gt_mean = torch.from_numpy(data['sample 0 mean']).cuda()
        # gt_logvar = torch.from_numpy(data['sample 0 logvar']).cuda()
        # sample_points=8192
        # sample_part_id = 0
        # start=0
        # how_many_each = 10
        # ids = list(range(64))
        sample_points = seg_mask.shape[1]
        how_many = codes.shape[0]
        codes = codes.unsqueeze(1).repeat_interleave(how_many_each, dim=1)
        # valid_id = valid_id[ids]
        # gt_mean = gt_mean[ids]
        # gt_logvar = gt_logvar[ids]
        print(codes.shape, valid_id.shape, gt_mean.shape, gt_logvar.shape)
        w = torch.randn(how_many*how_many_each, self.zdim).cuda()
        if self.use_flow:
            part_code = self.flow[sample_part_id](w, reverse=True).view(how_many, how_many_each, self.zdim)
        else:
            part_code = w.view(how_many, how_many_each, self.zdim)
        codes[..., sample_part_id] = part_code
        if fix_size:
            param_sample_num = 1
            selective_param_sample = False
        K = 100 
        min_noises = []
        min_means = []
        min_logvars = []
        for i in range(how_many):
            noise_latent = torch.randn(how_many_each* K, self.part_aligner.noise_dim).cuda()
            _part_code = codes[i][:, None].expand(-1, K, -1, -1).reshape(K*how_many_each, self.zdim, self.n_class)
            _gt_mean = gt_mean[i].reshape(1, 1, 3, self.n_class)
            _gt_logvar = gt_logvar[i].reshape(1, 1, 3, self.n_class)
            _valid_id = valid_id[i].reshape(1, self.n_class).expand(how_many_each * K, -1)
            mean, logvar = self.get_params_from_part_code(_part_code, _valid_id, noise=noise_latent)
            print(mean.shape, logvar.shape)
            mean, logvar = mean.reshape(how_many_each, K, 3, self.n_class), logvar.reshape(how_many_each, K, 3, self.n_class)
            if fix_size:
                fit_loss = (torch.cat([mean, logvar], dim=2) - torch.cat([_gt_mean, _gt_logvar], dim=2)) ** 2
                fit_loss = fit_loss.sum(2) * _valid_id.reshape(how_many_each, K, self.n_class)
                fit_loss[..., sample_part_id] = 0
                min_noise_idx = fit_loss.sum(-1).argmin(dim=1).int()
                min_noise_idx = min_noise_idx.unsqueeze(-1)
            else:
                if selective_param_sample:
                    (mean, logvar), min_noise_idx = self.subsample_params(mean, logvar, valid_id[:, 0], num=param_sample_num, return_ids=True)
                else:
                    min_noise_idx = torch.arange(K).unsqueeze(1).cuda() < param_sample_num
                    
            print(min_noise_idx.shape)
            min_noise = gather_operation(noise_latent.reshape(how_many_each, K, self.part_aligner.noise_dim).transpose(1,2).contiguous(), min_noise_idx).transpose(1,2).reshape(how_many_each, param_sample_num, self.part_aligner.noise_dim)
            min_mean = gather_operation(mean.reshape(how_many_each, K, -1).transpose(1,2).contiguous(), min_noise_idx).transpose(1,2).reshape(how_many_each, param_sample_num, 3, self.n_class)
            min_logvar = gather_operation(logvar.reshape(how_many_each, K, -1).transpose(1,2).contiguous(), min_noise_idx).transpose(1,2).reshape(how_many_each, param_sample_num, 3, self.n_class)
            min_noises.append(min_noise)
            min_means.append(min_mean)
            min_logvars.append(min_logvar)
        min_noises = torch.stack(min_noises, dim=0)
        min_means = torch.stack(min_means, dim=0)
        min_logvars = torch.stack(min_logvars, dim=0)
        print(min_noises.shape, min_means.shape, min_logvars.shape)
        assert codes.shape[-1] == self.n_class
        codes = codes.reshape(how_many * how_many_each, 1, self.zdim, self.n_class).repeat_interleave(param_sample_num, dim=1).reshape(how_many * how_many_each*param_sample_num, self.zdim, self.n_class)
        min_noises = min_noises.reshape(how_many * how_many_each * param_sample_num, self.part_aligner.noise_dim)
        min_means = min_means.reshape(how_many * how_many_each * param_sample_num, 3, self.n_class)
        min_logvars = min_logvars.reshape(how_many * how_many_each * param_sample_num, 3, self.n_class)
        valid_id = valid_id.reshape(how_many, 1, self.n_class).repeat_interleave(how_many_each*param_sample_num, dim=1).reshape(how_many_each * how_many*param_sample_num, self.n_class)
        seg_mask = seg_mask.reshape(how_many, 1, sample_points).expand(-1, how_many_each * param_sample_num, -1).reshape(-1, sample_points)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=min_means, variances=min_logvars)
        ctx = self.prepare_ctx(codes, min_means, min_logvars, anchor_assignments=seg_mask)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, valid_id.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_id, [codes, min_noises, min_means, min_logvars]
    
    def sample_with_one_fixed_latents(self):
        import pickle
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp655_5e-4_debugged_adamax_n50/val/sample_airplane_n50_encode_1000.pkl", "rb"))
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/sample_624_train_encode_1250.pkl", "rb"))
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_debugged_lamp_1e-3_n10_/val/sample_lamp_n10_4000.pkl", "rb"))
        data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/sample_chairs_from_lion_1250.pkl", "rb"))
        
        codes = torch.from_numpy(data['part_latents']).cuda()
        valid_id = torch.from_numpy(data['valid_id']).cuda()
        gt_mean = torch.from_numpy(data['sample 0 mean']).cuda()
        gt_logvar = torch.from_numpy(data['sample 0 logvar']).cuda()
        sample_points=2048
        fixed_part_id = 0
        # sample_ids = [9, 0, 20, 37, 53, 75, 81, 94]
        sample_ids = [34]
        how_many = len(sample_ids)
        how_many_each = 10
        codes = codes[sample_ids].reshape(how_many, 1, self.zdim, self.n_class).expand(-1, how_many_each, -1, -1).clone()
        gt_mean = gt_mean[sample_ids]
        gt_logvar = gt_logvar[sample_ids]
        valid_ids = torch.ones([how_many, how_many_each, self.n_class]).cuda()
        # valid_ids = torch.tensor([0, 1, 1, 1]).reshape(1, 1, self.n_class).cuda().repeat_interleave(how_many_each, dim=1)
        #setting arm to none
        valid_ids[:, :, 3] = 0
        # setting back to none
        # valid_ids[:, 27:30, 0] = 0
        print(codes.shape, gt_mean.shape, gt_logvar.shape)
        w = torch.randn(how_many*how_many_each, self.zdim).cuda()
        for i in range(self.n_class):
            if i == fixed_part_id: 
                continue
            codes[..., i] = self.flow[i](w, reverse=True).view(how_many, how_many_each, self.zdim)
        K = 100 
        keep_K = 10
        min_noises = []
        min_means = []
        min_logvars = []
        for i, s_id in enumerate(sample_ids):
            noise_latent = torch.randn(how_many_each* K, self.part_aligner.noise_dim).cuda()
            _part_code = codes[i][:, None].expand(-1, K, -1, -1).reshape(K*how_many_each, self.zdim, self.n_class)
            _gt_mean = gt_mean[i].reshape(1, 1, 3, self.n_class)
            _gt_logvar = gt_logvar[i].reshape(1, 1, 3, self.n_class)
            _valid_id = valid_ids[i].reshape(how_many_each, 1, self.n_class).expand(-1, K, -1).reshape(how_many_each*K, self.n_class)
            mean, logvar = self.get_params_from_part_code(_part_code, _valid_id, noise=noise_latent)
            print(mean.shape, logvar.shape)
            mean, logvar = mean.reshape(how_many_each, K, 3, self.n_class), logvar.reshape(how_many_each, K, 3, self.n_class)
            fit_loss = (torch.cat([mean, logvar], dim=2) - torch.cat([_gt_mean, _gt_logvar], dim=2)) ** 2
            fit_loss = fit_loss[:, :, 3:].sum(2)[..., fixed_part_id]
            print(fit_loss.shape)
            min_noise_idx = fit_loss.topk(keep_K, dim=1, largest=False)[1].int()
            print(min_noise_idx.shape)
            min_noise = gather_operation(noise_latent.reshape(how_many_each, K, self.part_aligner.noise_dim).transpose(1,2).contiguous(), min_noise_idx).transpose(1,2)
            min_mean = gather_operation(mean.reshape(how_many_each, K, -1).transpose(1,2).contiguous(), min_noise_idx).transpose(1,2).reshape(how_many_each, keep_K, 3, self.n_class)
            min_logvar = gather_operation(logvar.reshape(how_many_each, K, -1).transpose(1,2).contiguous(), min_noise_idx).transpose(1,2).reshape(how_many_each, keep_K, 3, self.n_class)
            min_noises.append(min_noise)
            min_means.append(min_mean)
            min_logvars.append(min_logvar)
        min_noises = torch.stack(min_noises, dim=0)
        min_means = torch.stack(min_means, dim=0)
        min_logvars = torch.stack(min_logvars, dim=0)
        print(min_noises.shape, min_means.shape, min_logvars.shape)
        assert codes.shape[-1] == self.n_class
        codes = codes.reshape(how_many * how_many_each, 1, self.zdim, self.n_class).expand(-1, keep_K, -1, -1).reshape(-1, self.zdim, self.n_class)
        min_noises = min_noises.reshape(how_many * how_many_each * keep_K, self.part_aligner.noise_dim)
        min_means = min_means.reshape(how_many * how_many_each * keep_K, 3, self.n_class)
        min_logvars = min_logvars.reshape(how_many * how_many_each * keep_K, 3, self.n_class)
        valid_ids = valid_ids.reshape(how_many, how_many_each,1 , self.n_class).repeat_interleave(keep_K, dim=2).reshape(how_many_each * how_many*keep_K, self.n_class)
        ids = torch.arange(self.n_class).cuda().unsqueeze(0) * valid_ids + valid_ids.argmax(dim=1, keepdim=True) * (1 - valid_ids)
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(how_many_each * how_many * keep_K, sample_points)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=min_means, variances=min_logvars)
        ctx = self.prepare_ctx(codes, min_means, min_logvars, anchor_assignments=seg_mask)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, valid_ids.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_ids, [codes, min_noises, min_means, min_logvars]
    
    def sample_with_given_latents(self):
        import pickle
        
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/part_sample_back_chairs_lion_1250.pkl", "rb"))
        # ref_data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp657_noise50/val/sample_car_1000.pkl", "rb"))
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_debugged_lamp_1e-3_n10_/val/lamp_fix_cap0_lamp_n10_4000.pkl", "rb"))
        # data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp655_5e-4_debugged_adamax_n50/val/tail_2322_airplane_n50_1000.pkl", "rb"))
        data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp657_noise50/val/param_variation_lamp_car_1000.pkl", "rb"))
        codes = torch.from_numpy(data['part_codes']).cuda()
        noises = torch.from_numpy(data['noises']).cuda()
        gt_mean = torch.from_numpy(data['means']).cuda()
        gt_logvar = torch.from_numpy(data['logvars']).cuda()
        seg_mask = torch.from_numpy(data['pred_seg_mask']).cuda()
        valid_ids = torch.zeros(codes.shape[0], self.n_class).cuda()
        for i in range(self.n_class):
            _valid = seg_mask == i 
            _valid = torch.any(_valid, dim=1).int()
            valid_ids[:, i] = _valid
        # ref_code = torch.from_numpy(ref_data['part_latents']).cuda()[33]
        sample_points=8192
        # sample_ids = [175]
        sample_ids = list(range(60, 70))
        # sample_ids = []
        # for i in ids:
        #     sample_ids = sample_ids + [i * 3, i * 3 + 1, i * 3 + 2]
        codes = codes[sample_ids]
        # codes[..., 0] = ref_code[..., 0]
        noises = noises[sample_ids]
        gt_mean = gt_mean[sample_ids]
        gt_logvar = gt_logvar[sample_ids]
        valid_ids = valid_ids[sample_ids]

        assert codes.shape[-1] == self.n_class
        seg_mask = torch.arange(self.n_class).cuda().repeat_interleave(torch.tensor([512, 512, 512, 8192 - 3 * 512]).cuda(), dim=0)[None].repeat_interleave(valid_ids.shape[0], dim=0).int()
        print(seg_mask.shape)
        # seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(-1, sample_points)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_mean, variances=gt_logvar)
        ctx = self.prepare_ctx(codes, gt_mean, gt_logvar, anchor_assignments=seg_mask)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, valid_ids.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_ids, [codes, noises, gt_mean, gt_logvar]
    def sample_variation_with_fixed_latents(self):
        import pickle
        data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp655_5e-4_debugged_adamax_n50/val/sample_airplane_n50_encode_1000.pkl", "rb"))
        codes = torch.from_numpy(data['part_latents']).cuda()
        # gt_mean = torch.from_numpy(data['means']).cuda()
        # gt_logvar = torch.from_numpy(data['logvars']).cuda()
        seg_mask = torch.from_numpy(data['pred_seg_mask']).cuda()
        valid_ids = torch.zeros(codes.shape[0], self.n_class).cuda()
        for i in range(self.n_class):
            _valid = seg_mask == i 
            _valid = torch.any(_valid, dim=1).int()
            valid_ids[:, i] = _valid
        
        sample_points=8192
        sample_ids = [6, 22]
        print(codes.shape, seg_mask.shape)
        means = []
        logvars = []
        noises = []
        for s_id in sample_ids:
            K = 1000 
            code = codes[s_id][None].repeat_interleave(K, dim=0)
            valid_id = valid_ids[s_id][None].repeat_interleave(K, dim=0)
            print(code.shape, valid_id.shape)
            noise_latent = torch.randn(K, self.part_aligner.noise_dim).cuda()
            mean, logvar = self.get_params_from_part_code(code, valid_id, noise=noise_latent)
            (mean, logvar), ids = self.subsample_params(mean.reshape(1, K, 3, self.n_class), logvar.reshape(1, K, 3, self.n_class), valid_id.reshape(1, K, self.n_class)[:, 0], num=50, return_ids=True)
            ids = ids[0]
            print(ids.shape)
            selected_noise = noise_latent[ids]
            print(mean.shape, logvar.shape, selected_noise.shape)
            means.append(mean)
            logvars.append(logvar)
            noises.append(selected_noise)
        means = torch.cat(means, dim=0)
        logvars = torch.cat(logvars, dim=0)
        noises = torch.cat(noises, dim=0)
        codes = codes[sample_ids].reshape(-1, 1, self.zdim, self.n_class).repeat_interleave(50, dim=1).reshape(-1, self.zdim, self.n_class)
        valid_ids = valid_ids[sample_ids].reshape(-1, 1, self.n_class).repeat_interleave(50, dim=1).reshape(-1, self.n_class)
        print(codes.shape, valid_ids.shape, means.shape, logvars.shape, noises.shape)
            
        ids = torch.arange(self.n_class).cuda().unsqueeze(0) * valid_ids + torch.argmax(valid_ids, dim=1, keepdim=True) * (1 - valid_ids)
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(-1, sample_points)
        print(seg_mask.shape)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=means, variances=logvars)
        ctx = self.prepare_ctx(codes, means, logvars, anchor_assignments=seg_mask)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, valid_ids.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_ids, [codes, noises, means, logvars]
    
    def part_swapping_with_fixed_latents(self):
        import pickle
        data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/encoded_variation_10_624_train_encode_1250.pkl", "rb"))
        ref_data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/sample_624_train_encode_1250.pkl", "rb"))
        ref_codes = torch.from_numpy(ref_data['part_latents']).cuda()
        codes = torch.from_numpy(data['part_codes']).cuda()
        noise_latents = torch.from_numpy(data['noises']).cuda()
        gt_mean = torch.from_numpy(data['means']).cuda()
        gt_logvar = torch.from_numpy(data['logvars']).cuda()
        seg_mask = torch.from_numpy(data['pred_seg_mask']).cuda()
        valid_ids = torch.zeros(codes.shape[0], self.n_class).cuda()
        for i in range(self.n_class):
            _valid = seg_mask == i 
            _valid = torch.any(_valid, dim=1).int()
            valid_ids[:, i] = _valid
        
        sample_points=2048
        start=0
        swap_ids = torch.tensor([0, 1, 0, 1]).cuda()
        ref_ids = [10, 16]
        codes = codes[start: start+50]
        noise_latents = noise_latents[start: start+50]
        gt_mean = gt_mean[start: start+50]
        gt_logvar = gt_logvar[start: start+50]
        valid_ids = valid_ids[start: start+50]
        print(codes.shape, gt_mean.shape, gt_logvar.shape, seg_mask.shape)
        means = []
        logvars = []
        noises = []
        swapped_codes = []
        swapped_valid_ids = []
        for s_id in ref_ids:
            K = 1000 
            ref_code = ref_codes[s_id][None]
            code = ref_code * swap_ids.unsqueeze(0).unsqueeze(0) + codes * (1 - swap_ids.unsqueeze(0).unsqueeze(0))
            valid_id = torch.ones_like(code)[:, 0]
            print(code.shape, valid_id.shape)
            mean, logvar = self.get_params_from_part_code(code, valid_id, noise=noise_latents)
            # (mean, logvar), ids = self.subsample_params(mean.reshape(1, K, 3, self.n_class), logvar.reshape(1, K, 3, self.n_class), valid_id.reshape(1, K, self.n_class)[:, 0], num=50, return_ids=True)
            # ids = ids[0]
            # print(ids.shape)
            # selected_noise = noise_latent[ids]
            # print(mean.shape, logvar.shape, selected_noise.shape)
            means.append(mean)
            logvars.append(logvar)
            noises.append(noise_latents)
            swapped_valid_ids.append(valid_id)
            swapped_codes.append(code)
        means = torch.cat(means, dim=0)
        logvars = torch.cat(logvars, dim=0)
        noises = torch.cat(noises, dim=0)
        swapped_valid_ids = torch.cat(swapped_valid_ids, dim=0)
        swapped_codes = torch.cat(swapped_codes, dim=0)
        print(swapped_codes.shape, swapped_valid_ids.shape, means.shape, logvars.shape, noises.shape)
            
        ids = torch.arange(self.n_class).cuda().unsqueeze(0) * swapped_valid_ids + torch.argmax(swapped_valid_ids, dim=1, keepdim=True) * (1 - swapped_valid_ids)
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(-1, sample_points)
        print(seg_mask.shape)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=means, variances=logvars)
        ctx = self.prepare_ctx(swapped_codes, means, logvars, anchor_assignments=seg_mask)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, swapped_valid_ids.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, swapped_valid_ids, [swapped_codes, noises, means, logvars]
    
    def part_interpolatioin_with_fixed_latents(self):
        import pickle
        data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/part_swapping_406_624_train_encode_1250.pkl", "rb"))
        ref_data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp624_n100_debugged/val/sample_624_train_encode_1250.pkl", "rb"))
        ref_codes = torch.from_numpy(ref_data['part_latents']).cuda()
        codes = torch.from_numpy(data['part_codes']).cuda()
        noise_latents = torch.from_numpy(data['noises']).cuda()
        gt_mean = torch.from_numpy(data['means']).cuda()
        gt_logvar = torch.from_numpy(data['logvars']).cuda()
        seg_mask = torch.from_numpy(data['pred_seg_mask']).cuda()
        valid_ids = torch.zeros(codes.shape[0], self.n_class).cuda()
        for i in range(self.n_class):
            _valid = seg_mask == i 
            _valid = torch.any(_valid, dim=1).int()
            valid_ids[:, i] = _valid
        
        sample_points=2048
        interpolate_part_id = 0
        param_ids = [4, 54]
        ref_ids = [5, 6, 12, 63, 115]
        codes = codes[param_ids]
        noise_latents = noise_latents[param_ids]
        gt_mean = gt_mean[param_ids]
        gt_logvar = gt_logvar[param_ids]
        valid_ids = valid_ids[param_ids]
        print(codes.shape, gt_mean.shape, gt_logvar.shape, seg_mask.shape)
        means = []
        logvars = []
        noises = []
        interpolated_codes = []
        interpolated_valid_ids = []
        for s_id in ref_ids:
            dx = torch.arange(0, 1.0, 0.1).cuda()
            ref_interpolate_part_code = ref_codes[s_id, :, interpolate_part_id]
            interpolate_part_code = codes[..., interpolate_part_id]
            part_code_diff = ref_interpolate_part_code[None] - interpolate_part_code
            interpolated_code = codes[:, None].expand(-1, 10, -1, -1).clone()
            interpolated_noise_latents = noise_latents[:, None].repeat_interleave(10, dim=1)
            interpolated_code[..., interpolate_part_id] = interpolate_part_code.reshape(-1, 1, self.zdim) + part_code_diff.reshape(-1, 1, self.zdim) * dx.reshape(1, 10, 1)
            interpolated_code = interpolated_code.reshape(-1, self.zdim, self.n_class)
            interpolated_noise_latents = interpolated_noise_latents.reshape(-1, self.part_aligner.noise_dim)
            valid_id = torch.ones_like(interpolated_code)[:, 0]
            
            print(interpolated_code.shape, valid_id.shape, interpolated_noise_latents.shape)
            mean, logvar = self.get_params_from_part_code(interpolated_code, valid_id, noise=interpolated_noise_latents)
            # (mean, logvar), ids = self.subsample_params(mean.reshape(1, K, 3, self.n_class), logvar.reshape(1, K, 3, self.n_class), valid_id.reshape(1, K, self.n_class)[:, 0], num=50, return_ids=True)
            # ids = ids[0]
            # print(ids.shape)
            # selected_noise = noise_latent[ids]
            # print(mean.shape, logvar.shape, selected_noise.shape)
            means.append(mean)
            logvars.append(logvar)
            noises.append(interpolated_noise_latents)
            interpolated_valid_ids.append(valid_id)
            interpolated_codes.append(interpolated_code)
        means = torch.cat(means, dim=0)
        logvars = torch.cat(logvars, dim=0)
        noises = torch.cat(noises, dim=0)
        interpolated_valid_ids = torch.cat(interpolated_valid_ids, dim=0)
        interpolated_codes = torch.cat(interpolated_codes, dim=0)
        print(interpolated_codes.shape, interpolated_valid_ids.shape, means.shape, logvars.shape, noises.shape)
            
        ids = torch.arange(self.n_class).cuda().unsqueeze(0) * interpolated_valid_ids + torch.argmax(interpolated_valid_ids, dim=1, keepdim=True) * (1 - interpolated_valid_ids)
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(-1, sample_points)
        print(seg_mask.shape)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=means, variances=logvars)
        ctx = self.prepare_ctx(interpolated_codes, means, logvars, anchor_assignments=seg_mask)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, interpolated_valid_ids.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, interpolated_valid_ids, [interpolated_codes, noises, means, logvars]
    
    @torch.no_grad()
    def interpolate_two_shape(self, xyz1, mask1, xyz2, mask2, sample_points, part_id, param_shift, param_scale, valid_id, param, mid_num=10):
        B = xyz1.shape[0]
        one_hot_mask1 = F.one_hot(mask1, num_classes=self.n_class)
        one_hot_mask2 = F.one_hot(mask2, num_classes=self.n_class)
        part_code_means1, part_code_logvars1 = self.get_part_code(xyz1, one_hot_mask1)
        part_code_means2, part_code_logvars2 = self.get_part_code(xyz2, one_hot_mask2)
        part_code1 = reparameterize_gaussian(mean=part_code_means1, logvar=part_code_logvars1).transpose(1,2)
        part_code2 = reparameterize_gaussian(mean=part_code_means2, logvar=part_code_logvars2).transpose(1,2)[..., part_id]
        noise = torch.randn(B, 1000, self.part_aligner.noise_dim).cuda()
        mean, logvar = self.get_params_from_part_code(part_code1.repeat_interleave(1000, dim=0), valid_id.repeat_interleave(1000, dim=0), noise=noise)
        print(mean.shape, logvar.shape)
        mean = mean.reshape(B, 1000, 3, 4) 
        logvar  = logvar.reshape(B, 1000, 3, 4) 
        print(torch.cat([mean, logvar], dim=2).shape)
        print(torch.cat([param_shift, 2 * torch.log(param_scale)], dim=1).shape)
        fit_loss = (torch.cat([mean, logvar], dim=2) - torch.cat([param_shift, 2 * torch.log(param_scale)], dim=1).unsqueeze(1)) ** 2
        print(fit_loss.shape, valid_id.shape)
        fit_loss = fit_loss.sum(1) * valid_id.unsqueeze(1) # [b K 4]
        fit_loss = fit_loss.sum(-1) / torch.sum(valid_id, dim=-1, keepdim=True)
        min_noise_idx = torch.argmin(fit_loss, dim=1)
        print(min_noise_idx.shape)
        noise = gather_operation(noise.transpose(1,2).contiguous(), min_noise_idx[:, None].int()).reshape(B, 1, self.part_aligner.noise_dim).expand(-1, mid_num, -1)
        K = mid_num
        dx = torch.linspace(0, 1, steps=K).reshape(1, -1, 1).cuda()
        interpolated_part_code = part_code1[..., part_id].unsqueeze(1) + (part_code2.unsqueeze(1) - part_code1[..., part_id].unsqueeze(1)) * dx
        print(interpolated_part_code.shape)
        part_code1 = part_code1.unsqueeze(1).repeat_interleave(K, dim=1)
        print(part_code1.shape)
        # noise = noise.unsqueeze(1).repeat_interleave(K, dim=1)
        valid_id = valid_id.unsqueeze(1).repeat_interleave(K, dim=1)
        part_code1[..., part_id] = interpolated_part_code
        part_code1 = part_code1.reshape(B*K, -1, 4)
        noise = noise.reshape(B*K,-1)
        valid_id = valid_id.reshape(B*K, 4)
        mean, logvar = self.get_params_from_part_code(part_code1, valid_id, noise=noise)
        mask1 = mask1.reshape(-1, mask1.shape[1], 1).expand(-1, -1, sample_points // mask1.shape[1]).reshape(-1, sample_points)
        anchor_assignments = mask1.repeat_interleave(K, dim=0).int()
        print(anchor_assignments.shape)
        anchor_per_point, logvar_per_point, _ = self.gather_all(anchor_assignments, anchors=mean, variances=logvar)
        variance_per_point = torch.exp(logvar_per_point)
        ctx = self.prepare_ctx(part_code1, mean, logvar, anchor_assignments=anchor_assignments)
        return anchor_per_point, ctx, variance_per_point, anchor_assignments, valid_id
        
    def sample_latents(self, sample_num, sample_points, device, fixed_id=None, valid_id=None, epoch=0, K=None, part_code=None, **kwargs):
        if part_code is None:
            part_code = torch.randn(sample_num, self.zdim, self.n_class).to(device) * math.sqrt(self.prior_var)
            if self.use_flow:
                part_code_ = []
                for i in range(self.n_class):
                    _part_code = part_code[..., i]
                    part_code_.append(self.flow[i](_part_code, reverse=True).view(sample_num, self.zdim))
                part_code = torch.stack(part_code_, dim=-1)
        if self.part_aligner and self.part_aligner.cimle:
            K = 10 if K is None else K
            if self.selective_noise_sampling or self.selective_noise_sampling_global:
                K = 100
            noise = torch.randn(sample_num*K, self.part_aligner.noise_dim).to(device)
            if self.part_aligner.cimle_start_epoch > epoch:
                noise = torch.zeros_like(noise)
        else:
            K=1
            noise=None
        fixed_codes = part_code[0].unsqueeze(0)
        fixed_valid_id = valid_id[0].unsqueeze(0)
        fixed_valid_id = (fixed_valid_id + fixed_id[None]).clamp(min=0, max=1)
        part_code = part_code * (1 - fixed_id).unsqueeze(0).unsqueeze(0) + fixed_id.unsqueeze(0).unsqueeze(0) * fixed_codes
        valid_id = valid_id * (1 - fixed_id).unsqueeze(0) + fixed_id.unsqueeze(0) * fixed_valid_id
        if noise is not None and torch.any(fixed_id == 1):
            noise = noise.reshape(sample_num, K, self.part_aligner.noise_dim)
            print(noise.shape)
            fixed_noise = noise[0].unsqueeze(0)
            noise = fixed_noise.expand(sample_num, -1, -1)
            noise = noise.reshape(sample_num* K, self.part_aligner.noise_dim)
        part_code = part_code.repeat_interleave(K, dim=0)
        
        if valid_id is None:
            valid_id = torch.ones(sample_num, self.n_class).to(part_code)
            
        valid_id = valid_id.repeat_interleave(K, dim=0)
        mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise=noise, **kwargs)
        if self.selective_noise_sampling:
            mean, logvar = self.subsample_params(mean.reshape(sample_num, K, 3, self.n_class), logvar.reshape(sample_num, K, 3, self.n_class), valid_id.reshape(sample_num, K, self.n_class)[:, 0], num=10)
            print(mean.shape)
            K = 10
            valid_id = valid_id.reshape(sample_num, 100, self.n_class)[:, :K].reshape(sample_num*K, self.n_class)
            part_code = part_code.reshape(sample_num, 100, -1, self.n_class)[:, :K].reshape(sample_num*K, -1, self.n_class)
        if self.selective_noise_sampling_global:
            (mean, logvar), selected_ids = self.subsample_params_global(mean, logvar, valid_id, num=sample_num*10)
            print(mean.shape, selected_ids.shape)
            valid_id = valid_id[selected_ids]
            part_code = part_code[selected_ids]
            K = 10
        print(K)
        # print(logvar.reshape(sample_num, 10, 3, self.n_class))
        assert part_code.shape[-1] == self.n_class
        # seg_mask = torch.arange(self.n_class).to(device).unsqueeze(0).repeat_interleave(torch.tensor([1024, 512, 512, 8192-2048]).cuda(), dim=1).repeat_interleave(sample_num*K, dim=0).int()
        ids = torch.arange(self.n_class).to(device).unsqueeze(0) * valid_id + torch.argmax(valid_id, dim=1).unsqueeze(1) * (1 - valid_id)
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(sample_num*K, sample_points)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=mean, variances=logvar + self.log_scale_var if logvar is not None else logvar)
        ctx = self.prepare_ctx(part_code, mean, logvar + self.log_scale_var if logvar is not None else logvar, anchor_assignments=seg_mask)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, valid_id.shape)
        # return ctx, mean_per_point, logvar_per_point, seg_mask, valid_id, [part_code, mean, logvar, noise]
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_id
    
    
    def get_prior_loss_part(self, part_code, mean, logvar, i, epoch=-1):
        B, C= part_code.shape 
        assert mean.shape[0] == logvar.shape[0]  == B
        entropy = gaussian_entropy(logvar=logvar.reshape(B, -1), dim=1)
        if self.use_flow:
            w, delta_log_pw = self.flow[i](part_code, torch.zeros([B, 1]).to(part_code), reverse=False)
            log_pw = gaussian_log_likelihood(w, 0, math.log(self.prior_var), dim=w.shape[1]).view(B, -1).sum(dim=1)   # (B, 1)
            log_p_part = log_pw - delta_log_pw.view(B)  # (B, 1)
        else:
            log_p_part = gaussian_log_likelihood(part_code, 0, math.log(self.prior_var), dim=C).sum(dim=-1)
        
        log_p_part = log_p_part.view(B)
        entropy = entropy.view(B)
        loss_prior = (- log_p_part - entropy)
        mmean = mean.mean(1)
        mlogvar = logvar.mean(1)
        mlog_p_part = log_p_part.mean()
        mentropy = entropy.mean()
        loss_dict = dict()
        if self.kl_weight_annealing and self.kl_weight_annealing_end_epoch > epoch:
            kl_weight = self.min_kl_weight + (self.kl_weight - self.min_kl_weight) * epoch / self.kl_weight_annealing_end_epoch
        else:
            kl_weight = self.kl_weight
        loss_dict['prior_loss'] = kl_weight * loss_prior.mean()
        loss_dict['kl_weight'] = torch.ones(1).cuda() * kl_weight
        loss_dict[f'log_p_part'] = mlog_p_part
        loss_dict[f'entropy'] = mentropy
        loss_dict[f'mean'] = mmean
        loss_dict[f'logvar'] = mlogvar
        return loss_dict
    def get_prior_loss(self, part_code, mean, logvar, valid_id, epoch=-1):
        B, C, M = part_code.shape 
        assert mean.shape[0] == logvar.shape[0] == valid_id.shape[0] == B
        assert mean.shape[1] == logvar.shape[1] == valid_id.shape[1] == M
        entropy = gaussian_entropy(logvar=logvar.reshape(B* self.n_class, -1), dim=1)
        if self.use_flow:
            log_p_part = torch.zeros(B, self.n_class).to(part_code)
            for i in range(self.n_class):
                _id = valid_id[:, i] == 1
                if _id.any():
                    _part_code = part_code[_id, :, i]
                    b = _part_code.shape[0]
                    # P(z), Prior probability, parameterized by the flow: z -> w.
                    # print(_part_code.abs().mean(1))
                    w, delta_log_pw = self.flow[i](_part_code, torch.zeros([b, 1]).to(part_code), reverse=False)
                    log_pw = gaussian_log_likelihood(w, 0, math.log(self.prior_var), dim=w.shape[1]).view(b, -1).sum(dim=1)   # (B, 1)
                    log_p_part[_id, i] = log_pw - delta_log_pw.view(b)  # (B, 1)
        else:
            log_p_part = gaussian_log_likelihood(part_code.transpose(1,2), 0, math.log(self.prior_var), dim=C).sum(dim=-1)
        
        log_p_part = log_p_part.view(B, self.n_class)
        entropy = entropy.view(B, self.n_class)
        loss_prior = ((- log_p_part - entropy) * valid_id).sum(1) / valid_id.sum(1)
        mmean = mean.mean(2).sum(0) / valid_id.sum(0)
        mlogvar = logvar.mean(2).sum(0) / valid_id.sum(0)
        mlog_p_part = (log_p_part * valid_id).sum(0) / valid_id.sum(0)
        mentropy = (entropy * valid_id).sum(0) / valid_id.sum(0)
        loss_dict = dict()
        if self.kl_weight_annealing and self.kl_weight_annealing_end_epoch > epoch:
            kl_weight = self.min_kl_weight + (self.kl_weight - self.min_kl_weight) * epoch / self.kl_weight_annealing_end_epoch
        else:
            kl_weight = self.kl_weight
        loss_dict['prior_loss'] = kl_weight * loss_prior.mean()
        loss_dict['kl_weight'] = torch.ones(1).cuda() * kl_weight
        for i in range(self.n_class):
            loss_dict[f'log_p_part_{i}'] = mlog_p_part[i]
            loss_dict[f'entropy_{i}'] = mentropy[i]
            # print(mlog_p_part[i])
            loss_dict[f'part_{i}_mean'] = mmean[i]
            loss_dict[f'part_{i}_logvar'] = mlogvar[i]
        return loss_dict
    
    def forward(self, pcds, device, noise=None, epoch=-1):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['ref_attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if noise is None:
            noise=pcds['noise'].to(device).unsqueeze(1)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        loss_dict = dict()
        B, C, N = ref.shape
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        if self.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)  # (B, F, M)
            prior_dict = self.get_prior_loss(part_code, part_code_means, part_code_logvars, valid_id, epoch=epoch)
            loss_dict.update(prior_dict)
        else:
            part_code = part_code_means.transpose(1,2)
        if self.normalize_part_code:
            part_code = F.normalize(part_code, p=2, dim=1)
        num_sample = noise.shape[1]
        noise = noise.reshape(B*num_sample, -1)
        if num_sample > 1:
            part_code, valid_id, seg_mask, ref, gt_shift, gt_var = map(lambda t: t.repeat_interleave(num_sample, dim=0), [part_code, valid_id, seg_mask, ref, gt_shift, gt_var])
        mean, logvar = self.get_params_from_part_code(part_code, valid_id, gt_mean=gt_shift, gt_var=gt_var, ref=ref, noise=noise)

        assert part_code.shape[-1]  == self.n_class
        
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        loss_dict['fit_loss'] = fit_loss
        if self.gt_param_annealing:
            assert epoch >=0, "must provide epoch as parameter when doing annealing"
            if epoch < self.gt_param_annealing_start_epoch:
                mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
                mean = gt_shift
                logvar = torch.log(gt_var)
            elif epoch < self.gt_param_annealing_end_epoch:
                gt_mean_per_point, gt_logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
                gt_mean = gt_shift
                gt_logvar = torch.log(gt_var)
                gt_prob = epoch / (self.gt_param_annealing_end_epoch - self.gt_param_annealing_start_epoch)
                gt_idx = (torch.randn(B*num_sample) >= gt_prob).to(part_code).reshape(-1, 1, 1)
                mean_per_point = mean_per_point * (1 - gt_idx) + gt_idx * gt_mean_per_point
                logvar_per_point = logvar_per_point * (1 - gt_idx) + gt_idx * gt_logvar_per_point
                mean = mean * (1 - gt_idx) + gt_idx * gt_mean
                logvar =  logvar * (1 - gt_idx) + gt_idx * gt_logvar
        if self.use_gt_params_in_training:
            mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
            mean = gt_shift
            logvar = torch.log(gt_var)
        ctx = self.prepare_ctx( part_code, mean, logvar, anchor_assignments=seg_mask, z=None)
        
        return ctx, mean_per_point, logvar_per_point + self.log_scale_var, flag_per_point, loss_dict, [part_code, mean, logvar, noise]



@ENCODERS.register_module()
class PartEncoderForPartnet(PartEncoder):
    def __init__(
        self, 
        **kwargs
        ):
        super().__init__(**kwargs)


    def forward(self, pcds, device):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        loss_dict = dict()
        B, C, N = ref.shape
        part_code = self.get_part_code(input, seg_flag)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, gt_mean=gt_shift, gt_var=gt_var, ref=ref)
        
        assert part_code.shape[-1]  == self.n_class
        
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=None)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        loss_dict['fit_loss'] = self.fit_loss_weight*fit_loss

        ctx = self.prepare_ctx(part_code, mean, logvar, anchor_assignments=seg_mask, z=z)
        
        return ctx, mean_per_point, logvar_per_point + self.log_scale_var, flag_per_point, loss_dict





@ENCODERS.register_module()
class PartEncoderForTransformerDecoder(PartEncoder):
    def __init__(
        self, 
        **kwargs
        ):
        super().__init__(**kwargs)


    def prepare_ctx(self, part_code, mean, logvar, **kwargs):
        ctx = []
        if self.include_part_code:
            ctx.append(part_code)
        if self.include_params:
            params = torch.cat([mean, torch.exp(logvar + self.log_scale_var)], dim=1)
            if self.detach_params_in_ctx:
                params = params.detach()
            ctx.append(params)
        return ctx


@ENCODERS.register_module()
class PartEncodercVAE(PartEncoderForTransformerDecoder):
    def __init__(
        self, 
        cvae_kl_weight=0.1,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.cvae_encoder=MLP([self.zdim*(1 + self.n_class), self.zdim, self.zdim, self.part_aligner.noise_dim*2], use_linear=True)
        assert self.encode_ref
        self.cvae_kl_weight=cvae_kl_weight
        
    def get_params_from_part_code(self, part_code, valid_id, gt_mean=None, gt_var=None, ref=None, noise=None, **kwargs):
        B = part_code.shape[0]
        if self.training:
            global_feat = self.ref_encoder(ref.transpose(1,2)).reshape(B, -1)
            cond = torch.cat([part_code.reshape(B, -1), global_feat], dim=-1)
            noise_mean, noise_logvar = torch.chunk(self.cvae_encoder(cond), 2, dim=-1)
            noise = reparameterize_gaussian(mean=noise_mean, logvar=noise_logvar)
            entropy = gaussian_entropy(logvar=noise_logvar, dim=1)
            log_p_part = gaussian_log_likelihood(noise, 0, 1.0, dim=self.part_aligner.noise_dim).sum(dim=-1)
            loss_prior_cvae = (- log_p_part - entropy).mean()
        else:
            loss_prior_cvae = torch.zeros(1).to(part_code)
        
        mean, logvar = self.part_aligner(part_code, valid_id, noise=noise, **kwargs)
        return  mean, logvar, loss_prior_cvae
    def sample_noise(self, pcds, device, num):
        input = pcds['input'].to(device)
        B, N, C = input.shape
        return torch.randn(B, num, self.part_aligner.noise_dim).to(device), None
    
    def forward(self, pcds, device, noise=None, epoch=-1):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['ref_attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if noise is None:
            noise=pcds['noise'].to(device).unsqueeze(1)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        loss_dict = dict()
        B, C, N = ref.shape
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        if self.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)  # (B, F, M)
            prior_dict = self.get_prior_loss(part_code, part_code_means, part_code_logvars, valid_id, epoch=epoch)
            loss_dict.update(prior_dict)
        else:
            part_code = part_code_means.transpose(1,2)
        if self.normalize_part_code:
            part_code = F.normalize(part_code, p=2, dim=1)
        num_sample = noise.shape[1]
        noise = noise.reshape(B*num_sample, -1)
        if num_sample > 1:
            part_code, valid_id, seg_mask, ref, gt_shift, gt_var = map(lambda t: t.repeat_interleave(num_sample, dim=0), [part_code, valid_id, seg_mask, ref, gt_shift, gt_var])
        mean, logvar, cvae_loss = self.get_params_from_part_code(part_code, valid_id, gt_mean=gt_shift, gt_var=gt_var, ref=ref, noise=noise)
        assert part_code.shape[-1]  == self.n_class
        
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        loss_dict['fit_loss'] = fit_loss
        loss_dict['cvae_loss'] = cvae_loss * self.cvae_kl_weight
        if self.gt_param_annealing:
            assert epoch >=0, "must provide epoch as parameter when doing annealing"
            if epoch < self.gt_param_annealing_start_epoch:
                mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
                mean = gt_shift
                logvar = torch.log(gt_var)
            elif epoch < self.gt_param_annealing_end_epoch:
                gt_mean_per_point, gt_logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
                gt_mean = gt_shift
                gt_logvar = torch.log(gt_var)
                gt_prob = epoch / (self.gt_param_annealing_end_epoch - self.gt_param_annealing_start_epoch)
                gt_idx = (torch.randn(B*num_sample) >= gt_prob).to(part_code).reshape(-1, 1, 1)
                mean_per_point = mean_per_point * (1 - gt_idx) + gt_idx * gt_mean_per_point
                logvar_per_point = logvar_per_point * (1 - gt_idx) + gt_idx * gt_logvar_per_point
                mean = mean * (1 - gt_idx) + gt_idx * gt_mean
                logvar =  logvar * (1 - gt_idx) + gt_idx * gt_logvar
        if self.use_gt_params_in_training:
            mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
            mean = gt_shift
            logvar = torch.log(gt_var)
        ctx = self.prepare_ctx( part_code, mean, logvar, anchor_assignments=seg_mask, z=None)
        
        return ctx, mean_per_point, logvar_per_point + self.log_scale_var, flag_per_point, loss_dict
    
@ENCODERS.register_module()
class PartEncodercVAE2(PartEncoderForTransformerDecoder):
    def __init__(
        self, 
        cvae_kl_weight=0.1,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.cvae_encoder=MLP([self.zdim*(self.n_class), self.zdim, self.zdim, self.part_aligner.noise_dim*2], use_linear=True)
        self.cvae_kl_weight=cvae_kl_weight
        
    def get_params_from_part_code(self, part_code, valid_id, gt_mean=None, gt_var=None, ref=None, noise=None, **kwargs):
        B = part_code.shape[0]
        if self.training:
            noise_mean, noise_logvar = torch.chunk(self.cvae_encoder(part_code.reshape(B, -1)), 2, dim=-1)
            noise = reparameterize_gaussian(mean=noise_mean, logvar=noise_logvar)
            entropy = gaussian_entropy(logvar=noise_logvar, dim=1)
            log_p_part = gaussian_log_likelihood(noise, 0, 1.0, dim=self.part_aligner.noise_dim).sum(dim=-1)
            loss_prior_cvae = (- log_p_part - entropy).mean()
        else:
            loss_prior_cvae = torch.zeros(1).to(part_code)
        
        mean, logvar = self.part_aligner(part_code, valid_id, noise=noise, **kwargs)
        return  mean, logvar, loss_prior_cvae
    def sample_noise(self, pcds, device, num):
        input = pcds['input'].to(device)
        B, N, C = input.shape
        return torch.randn(B, num, self.part_aligner.noise_dim).to(device), None
    
    def forward(self, pcds, device, noise=None, epoch=-1):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['ref_attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if noise is None:
            noise=pcds['noise'].to(device).unsqueeze(1)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        loss_dict = dict()
        B, C, N = ref.shape
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        if self.gen:
            part_code = reparameterize_gaussian(mean=part_code_means, logvar=part_code_logvars).transpose(1,2)  # (B, F, M)
            prior_dict = self.get_prior_loss(part_code, part_code_means, part_code_logvars, valid_id, epoch=epoch)
            loss_dict.update(prior_dict)
        else:
            part_code = part_code_means.transpose(1,2)
        if self.normalize_part_code:
            part_code = F.normalize(part_code, p=2, dim=1)
        num_sample = noise.shape[1]
        noise = noise.reshape(B*num_sample, -1)
        if num_sample > 1:
            part_code, valid_id, seg_mask, ref, gt_shift, gt_var = map(lambda t: t.repeat_interleave(num_sample, dim=0), [part_code, valid_id, seg_mask, ref, gt_shift, gt_var])
        mean, logvar, cvae_loss = self.get_params_from_part_code(part_code, valid_id, gt_mean=gt_shift, gt_var=gt_var, ref=ref, noise=noise)
        assert part_code.shape[-1]  == self.n_class
        
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        loss_dict['fit_loss'] = fit_loss
        loss_dict['cvae_loss'] = cvae_loss * self.cvae_kl_weight
        if self.gt_param_annealing:
            assert epoch >=0, "must provide epoch as parameter when doing annealing"
            if epoch < self.gt_param_annealing_start_epoch:
                mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
                mean = gt_shift
                logvar = torch.log(gt_var)
            elif epoch < self.gt_param_annealing_end_epoch:
                gt_mean_per_point, gt_logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
                gt_mean = gt_shift
                gt_logvar = torch.log(gt_var)
                gt_prob = epoch / (self.gt_param_annealing_end_epoch - self.gt_param_annealing_start_epoch)
                gt_idx = (torch.randn(B*num_sample) >= gt_prob).to(part_code).reshape(-1, 1, 1)
                mean_per_point = mean_per_point * (1 - gt_idx) + gt_idx * gt_mean_per_point
                logvar_per_point = logvar_per_point * (1 - gt_idx) + gt_idx * gt_logvar_per_point
                mean = mean * (1 - gt_idx) + gt_idx * gt_mean
                logvar =  logvar * (1 - gt_idx) + gt_idx * gt_logvar
        if self.use_gt_params_in_training:
            mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
            mean = gt_shift
            logvar = torch.log(gt_var)
        ctx = self.prepare_ctx( part_code, mean, logvar, anchor_assignments=seg_mask, z=None)
        
        return ctx, mean_per_point, logvar_per_point + self.log_scale_var, flag_per_point, loss_dict
    
@ENCODERS.register_module()
class PartEncoderVAE(PartEncodercVAE):
    def __init__(
        self, 
        **kwargs
        ):
        super().__init__(**kwargs)
        self.ref_encoder = PointNetVAEBase(zdim=self.part_aligner.noise_dim)        
        
    def get_params_from_part_code(self, part_code, valid_id, gt_mean=None, gt_var=None, ref=None, noise=None, **kwargs):
        B = part_code.shape[0]
        if self.training:
            noise_mean, noise_logvar = self.ref_encoder(ref.transpose(1,2))
            noise = reparameterize_gaussian(mean=noise_mean, logvar=noise_logvar)
            entropy = gaussian_entropy(logvar=noise_logvar, dim=1)
            log_p_part = gaussian_log_likelihood(noise, 0, 1.0, dim=self.part_aligner.noise_dim).sum(dim=-1)
            loss_prior_cvae = (- log_p_part - entropy).mean()
        else:
            loss_prior_cvae = torch.zeros(1).to(part_code)
        
        mean, logvar = self.part_aligner(part_code, valid_id, noise=noise, **kwargs)

        return  mean, logvar, loss_prior_cvae
    
    

@ENCODERS.register_module()
class PartEncoderForPartnetAndTransformerDecoder(PartEncoderForPartnet):
    def __init__(
        self, 
        **kwargs
        ):
        super().__init__(**kwargs)
    
    def prepare_ctx(self, part_code, mean, logvar, **kwargs):
        ctx = []
        if self.include_part_code:
            ctx.append(part_code)
        if self.include_params:
            ctx.append(torch.cat([mean, torch.exp(logvar + self.log_scale_var)], dim=1))
        return ctx
    
    
@ENCODERS.register_module()
class PartEncoderCIMLE(PartEncoder):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
    
    def get_params_from_part_code(self, part_code, valid_id, noise):
        mean, logvar = self.part_aligner(part_code, valid_id, noise)
        l = [mean]
        if self.include_var:
            l.append(logvar)
        else:
            logvar = None
        z, part_code = self.mixer(part_code, l, valid_id)
        return z, part_code, mean, logvar
    
    def sample_noise(self, pcds, device, num):
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        B, N, C = input.shape
        part_code = self.get_part_code(torch.cat([input, seg_flag], dim=-1))
        noise = torch.randn(B*num, self.part_aligner.noise_dim).to(device)
        part_code = part_code.repeat_interleave(num, dim=0)
        valid_id = valid_id.repeat_interleave(num, dim=0)
        seg_mask = seg_mask.repeat_interleave(num, dim=0)
        ref = ref.repeat_interleave(num, dim=0)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise)
        assert part_code.shape[-1] == self.n_class
        mean_per_point, logvar_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = gaussian_log_likelihood(ref, mean_per_point, logvar_per_point).mean(dim=(1,2))
        id = fit_loss.reshape(B, num).max(1)[1]
        return noise.reshape(B, num, -1), id
    def forward(self, pcds, device, noise=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        if noise is None:
            noise=pcds['noise'].to(device).unsqueeze(1)
        B, C, N = ref.shape
        part_code = self.get_part_code(torch.cat([input, seg_flag], dim=-1))
        num_sample=noise.shape[1]
        noise = noise.reshape(B*num_sample, -1)
        part_code = part_code.repeat_interleave(num_sample, dim=0)
        valid_id = valid_id.repeat_interleave(num_sample, dim=0)
        seg_mask = seg_mask.repeat_interleave(num_sample, dim=0)
        ref = ref.repeat_interleave(num_sample, dim=0)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise)
        assert part_code.shape[-1] == self.n_class
        
        mean_per_point, logvar_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        if self.training and self.part_aligner is not None:
            fit_loss = -gaussian_log_likelihood(ref, mean_per_point, logvar_per_point).mean()
        else:
            fit_loss = torch.zeros([1]).to(device)

        ctx = self.prepare_ctx(part_code, mean, logvar, anchor_assignments=seg_mask, z=z)
        
        return ctx, mean_per_point, logvar_per_point, dict(fit_loss=self.fit_loss_weight*fit_loss)


@ENCODERS.register_module()
class PartEncoderForTransformerDecoderCIMLE(PartEncoderForTransformerDecoder):
    def __init__(
        self, 
        **kwargs
        ):
        super().__init__(**kwargs)

    def get_params_from_part_code(self, part_code, valid_id, noise):
        mean, logvar = self.part_aligner(part_code, valid_id, noise)
        l = [mean]
        if self.include_var:
            l.append(logvar)
        else:
            logvar = None
        z, part_code = self.mixer(part_code, l, valid_id)
        return z, part_code, mean, logvar
    
    def sample_noise(self, pcds, device, num):
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        B, N, C = input.shape
        part_code = self.get_part_code(torch.cat([input, seg_flag], dim=-1))
        noise = torch.randn(B*num, self.part_aligner.noise_dim).to(device)
        part_code = part_code.repeat_interleave(num, dim=0)
        valid_id = valid_id.repeat_interleave(num, dim=0)
        seg_mask = seg_mask.repeat_interleave(num, dim=0)
        ref = ref.repeat_interleave(num, dim=0)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise)
        assert part_code.shape[-1] == self.n_class
        mean_per_point, logvar_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = gaussian_log_likelihood(ref, mean_per_point, logvar_per_point).mean(dim=(1,2))
        id = fit_loss.reshape(B, num).max(1)[1]
        return noise.reshape(B, num, -1), id
    def forward(self, pcds, device, noise=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        if noise is None:
            noise=pcds['noise'].to(device).unsqueeze(1)
        B, C, N = ref.shape
        part_code = self.get_part_code(torch.cat([input, seg_flag], dim=-1))
        num_sample=noise.shape[1]
        noise = noise.reshape(B*num_sample, -1)
        part_code = part_code.repeat_interleave(num_sample, dim=0)
        valid_id = valid_id.repeat_interleave(num_sample, dim=0)
        seg_mask = seg_mask.repeat_interleave(num_sample, dim=0)
        ref = ref.repeat_interleave(num_sample, dim=0)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise)
        assert part_code.shape[-1] == self.n_class
        
        mean_per_point, logvar_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        if self.training and self.part_aligner is not None:
            fit_loss = -gaussian_log_likelihood(ref, mean_per_point, logvar_per_point).mean()
        else:
            fit_loss = torch.zeros([1]).to(device)

        ctx = self.prepare_ctx(part_code, mean, logvar)
        
        return ctx, mean_per_point, logvar_per_point, dict(fit_loss=self.fit_loss_weight*fit_loss)
    
@ENCODERS.register_module()
class PartEncoderWithKLLoss(PartEncoder):
    def __init__(self, 
                 kl_weight=0.001, 
                 use_flow=False,
                 latent_flow_depth=14, 
                 latent_flow_hidden_dim=256, 
                 use_gt_params_in_training=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_gt_params_in_training=use_gt_params_in_training
        self.kl_weight=kl_weight
        self.use_flow=use_flow 
        if use_flow:
            self.flow = nn.ModuleList([build_latent_flow(latent_flow_depth, latent_flow_hidden_dim, self.encoder.zdim) for _ in range(self.n_class)])
        self.zdim = self.encoder.zdim

    def get_part_code(self, input, seg_flag):
        return self.encoder(torch.cat([input, seg_flag], dim=-1))
    
    def sample_noise(self, pcds, device, num):
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        B, N, C = input.shape
        
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        part_code = reparameterize_gaussian(mean=part_code_means.reshape(B* self.n_class, -1), logvar=part_code_logvars.reshape(B* self.n_class, -1)).reshape(B, self.n_class, -1).transpose(1,2)  # (B, F)
        
        noise = torch.randn(B*num, self.part_aligner.noise_dim).to(device)
        part_code, valid_id, seg_mask, ref, gt_shift, gt_var = map(lambda t: t.repeat_interleave(num, dim=0), [part_code, valid_id, seg_mask, ref, gt_shift, gt_var])
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise=noise, gt_mean=gt_shift, gt_var=gt_var, ref=ref)
        assert part_code.shape[-1] == self.n_class
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        id = fit_loss.reshape(B, num).min(1)[1]
        return noise.reshape(B, num, -1), id
    
    def cimle_sample_latents(self, sample_num, sample_points, device, fixed_id=None, valid_id=None, **kwargs):
        part_codes = torch.randn(sample_num, self.zdim, self.n_class).to(device)
        noise = torch.randn(sample_num*10, self.part_aligner.noise_dim).to(device)
        fixed_codes = part_codes[0].unsqueeze(0)
        part_codes = part_codes * (1 - fixed_id).unsqueeze(0).unsqueeze(0) + fixed_id.unsqueeze(0).unsqueeze(0) * fixed_codes
        part_codes = part_codes.repeat_interleave(10, dim=0)
        if self.use_flow:
            part_codes_ = []
            for i in range(self.n_class):
                _part_code = part_codes[..., i]
                part_codes_.append(self.flow[i](_part_code, reverse=True).view(sample_num, self.n_class, self.zdim))
            part_codes = torch.stack(_part_code, dim=-1)
        if valid_id is None:
            valid_id = torch.ones(sample_num, self.n_class).to(part_codes)
        valid_id = valid_id.repeat_interleave(10, dim=0)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_codes, valid_id, noise=noise, **kwargs)
        # print(logvar.reshape(sample_num, 10, 3, self.n_class))
        assert part_code.shape[-1] == self.n_class
        ids = torch.arange(self.n_class).to(device).unsqueeze(0) * valid_id
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(sample_num*10, sample_points)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=mean, variances=logvar + self.log_scale_var)
        ctx = self.prepare_ctx(part_codes, mean, logvar + self.log_scale_var)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, valid_id.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_id
    
    
    def sample_latents(self, sample_num, sample_points, device, fixed_id=None, valid_id=None, **kwargs):
        part_codes = torch.randn(sample_num, self.zdim, self.n_class).to(device)
        fixed_codes = part_codes[0].unsqueeze(0)
        part_codes = part_codes * (1 - fixed_id).unsqueeze(0).unsqueeze(0) + fixed_id.unsqueeze(0).unsqueeze(0) * fixed_codes
        if self.use_flow:
            part_codes_ = []
            for i in range(self.n_class):
                _part_code = part_codes[..., i]
                part_codes_.append(self.flow[i](_part_code, reverse=True).view(sample_num, self.zdim))
            part_codes = torch.stack(part_codes_, dim=-1)
        if valid_id is None:
            valid_id = torch.ones(sample_num, self.n_class).to(part_codes)

        z, part_code, mean, logvar = self.get_params_from_part_code(part_codes, valid_id, **kwargs)
        
        assert part_code.shape[-1] == self.n_class
        ids = torch.arange(self.n_class).to(device).unsqueeze(0) * valid_id
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(sample_num, sample_points)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=mean, variances=logvar + self.log_scale_var)
        ctx = self.prepare_ctx(part_codes, mean, logvar + self.log_scale_var, anchor_assignments=seg_mask, z=z)
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_id
    
    def get_prior_loss(self, part_code, mean, logvar, valid_id):
        B, C, M = part_code.shape 
        assert mean.shape[0] == logvar.shape[0] == valid_id.shape[0] == B
        assert mean.shape[1] == logvar.shape[1] == valid_id.shape[1] == M
        entropy = gaussian_entropy(logvar=logvar.reshape(B* self.n_class, -1), dim=1)
        if self.use_flow:
            log_p_part = []
            for i in range(self.n_class):
                _part_code = part_code[..., i]
                # P(z), Prior probability, parameterized by the flow: z -> w.
                w, delta_log_pw = self.flow[i](_part_code, torch.zeros([B, 1]).to(part_code), reverse=False)
                log_pw = standard_normal_logprob(w).view(B, -1).sum(dim=1)   # (B, 1)
                log_p_part.append(log_pw - delta_log_pw.view(B))  # (B, 1)
            log_p_part = torch.stack(log_p_part, dim=1)
        else:
            log_p_part = standard_normal_logprob(part_code.transpose(1,2).reshape(B*self.n_class, -1)).sum(dim=1)

        log_p_part = log_p_part.view(B, self.n_class)
        entropy = entropy.view(B, self.n_class)
        loss_prior = ((- log_p_part - entropy) * valid_id).sum(1) / valid_id.sum(1)
        mmean = mean.mean(2).sum(0) / valid_id.sum(0)
        mlogvar = logvar.mean(2).sum(0) / valid_id.sum(0)
        return loss_prior, mmean, mlogvar
    
    def forward(self, pcds, device, noise=None, epoch=-1):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if noise is None:
            noise=pcds['noise'].to(device).unsqueeze(1)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        loss_dict = dict()
        B, C, N = ref.shape
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        part_code = reparameterize_gaussian(mean=part_code_means.reshape(B* self.n_class, -1), logvar=part_code_logvars.reshape(B* self.n_class, -1)).reshape(B, self.n_class, -1).transpose(1,2)  # (B, F)
        
        loss_prior, mmean, mlogvar = self.get_prior_loss(part_code, part_code_means, part_code_logvars, valid_id)
        loss_dict['prior_loss'] = self.kl_weight * loss_prior.mean()
        for i in range(self.n_class):
            loss_dict[f'part_{i}_mean'] = mmean[i]
            loss_dict[f'part_{i}_logvar'] = mlogvar[i]
        
        num_sample=noise.shape[1]
        noise = noise.reshape(B*num_sample, -1)
        if num_sample > 1:
            part_code, valid_id, seg_mask, ref, gt_shift, gt_var = map(lambda t: t.repeat_interleave(num_sample, dim=0), [part_code, valid_id, seg_mask, ref, gt_shift, gt_var])
        
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, gt_mean=gt_shift, gt_var=gt_var, ref=ref, noise=noise)
        assert part_code.shape[-1] == self.n_class
        
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        loss_dict['fit_loss'] = self.fit_loss_weight * fit_loss
        if self.use_gt_params_in_training:
            mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=gt_shift, variances=torch.log(gt_var), valid_id=None)
            mean = gt_shift
            logvar = torch.log(gt_var)
        ctx = self.prepare_ctx(part_code, mean, logvar + self.log_scale_var, anchor_assignments=seg_mask, z=z)
        # print(loss_dict)
        
        return ctx, mean_per_point, logvar_per_point + self.log_scale_var, flag_per_point, loss_dict
    

@ENCODERS.register_module()
class PartEncoderForTransformerDecoderKLLoss(PartEncoderWithKLLoss):
    def __init__(
        self, 
        include_std=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.include_std=include_std
    
    def prepare_ctx(self, part_code, mean, logvar, **kwargs):
        ctx = []
        if self.include_part_code:
            ctx.append(part_code)
        if self.include_params:
            if not self.include_std:
                ctx.append(torch.cat([mean, torch.exp(logvar)], dim=1))
            else:
                ctx.append(torch.cat([mean, torch.exp(0.5 * logvar)], dim=1))
                
        return ctx


@ENCODERS.register_module()
class PartEncoderForTransformerDecoderKLLossCIMLE(PartEncoderForTransformerDecoderKLLoss):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(**kwargs)


    def sample_noise(self, pcds, device, num):
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        B, N, C = input.shape
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        part_code = reparameterize_gaussian(mean=part_code_means.reshape(B* self.n_class, -1), logvar=part_code_logvars.reshape(B* self.n_class, -1)).reshape(B, self.n_class, -1).transpose(1,2)  # (B, F)
        noise = torch.randn(B*num, self.part_aligner.noise_dim).to(device)
        part_code = part_code.repeat_interleave(num, dim=0)
        valid_id = valid_id.repeat_interleave(num, dim=0)
        seg_mask = seg_mask.repeat_interleave(num, dim=0)
        ref = ref.repeat_interleave(num, dim=0)
        gt_shift = gt_shift.repeat_interleave(num, dim=0)
        gt_var = gt_var.repeat_interleave(num, dim=0)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise=noise)
        assert part_code.shape[-1] == self.n_class
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        id = fit_loss.reshape(B, num).min(1)[1]
        return noise.reshape(B, num, -1), id
    
    def cimle_sample_latents(self, sample_num, sample_points, device, fixed_id=None, valid_id=None, **kwargs):
        part_codes = torch.randn(sample_num, self.zdim, self.n_class).to(device)
        noise = torch.randn(sample_num*10, self.part_aligner.noise_dim).to(device)
        fixed_codes = part_codes[0].unsqueeze(0)
        part_codes = part_codes * (1 - fixed_id).unsqueeze(0).unsqueeze(0) + fixed_id.unsqueeze(0).unsqueeze(0) * fixed_codes
        part_codes = part_codes.repeat_interleave(10, dim=0)
        if self.use_flow:
            part_codes_ = []
            for i in range(self.n_class):
                _part_code = part_codes[..., i]
                part_codes_.append(self.flow[i](_part_code, reverse=True).view(sample_num, self.n_class, self.zdim))
            part_codes = torch.stack(_part_code, dim=-1)
        if valid_id is None:
            valid_id = torch.ones(sample_num, self.n_class).to(part_codes)
        valid_id = valid_id.repeat_interleave(10, dim=0)
        z, part_code, mean, logvar = self.get_params_from_part_code(part_codes, valid_id, noise=noise, **kwargs)
        # print(logvar.reshape(sample_num, 10, 3, self.n_class))
        assert part_code.shape[-1] == self.n_class
        ids = torch.arange(self.n_class).to(device).unsqueeze(0) * valid_id
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(sample_num*10, sample_points)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=mean, variances=logvar + self.log_scale_var)
        ctx = self.prepare_ctx(part_codes, mean, logvar + self.log_scale_var)
        print(mean_per_point.shape, logvar_per_point.shape, seg_mask.shape, valid_id.shape)
        return ctx, mean_per_point, logvar_per_point, seg_mask, valid_id
    
    
    def forward(self, pcds, device, noise=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        input = pcds['input'].to(device)
        valid_id = pcds['present'].to(device)
        ref = pcds['ref'].to(device).transpose(1,2)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['ref_attn_map'].to(device)
        gt_shift = pcds.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = pcds.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device)
        if not self.origin_scale:
            gt_var = gt_var ** 2
        loss_dict = dict()
        B, C, N = ref.shape
        if noise is None:
            noise=pcds['noise'].to(device).unsqueeze(1)
            
        part_code_means, part_code_logvars = self.get_part_code(input, seg_flag)
        assert part_code_means.shape[1] == part_code_logvars.shape[1] == self.n_class
        part_code = reparameterize_gaussian(mean=part_code_means.reshape(B* self.n_class, -1), logvar=part_code_logvars.reshape(B* self.n_class, -1)).reshape(B, self.n_class, -1).transpose(1,2)  # (B, F)
        
        num_sample=noise.shape[1]
        noise = noise.reshape(B*num_sample, -1)
        if num_sample > 1:
            part_code_means = part_code_means.repeat_interleave(num_sample, dim=0)
            part_code_logvars = part_code_logvars.repeat_interleave(num_sample, dim=0)
            part_code = part_code.repeat_interleave(num_sample, dim=0)
            valid_id = valid_id.repeat_interleave(num_sample, dim=0)
            seg_mask = seg_mask.repeat_interleave(num_sample, dim=0)
            ref = ref.repeat_interleave(num_sample, dim=0)
            gt_shift = gt_shift.repeat_interleave(num_sample, dim=0)
            gt_var = gt_var.repeat_interleave(num_sample, dim=0)
            
        loss_prior = self.get_prior_loss(part_code, part_code_means, part_code_logvars, valid_id)
        loss_dict['prior_loss'] = self.kl_weight * loss_prior
        
        z, part_code, mean, logvar = self.get_params_from_part_code(part_code, valid_id, noise=noise)
        assert part_code.shape[-1] == self.n_class
        
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        fit_loss = self.get_fit_loss(ref, mean, logvar, valid_id, mean_per_point, logvar_per_point, flag_per_point, gt_shift, gt_var, seg_mask)
        loss_dict['fit_loss'] = self.fit_loss_weight * fit_loss
        ctx = self.prepare_ctx(part_code, mean, logvar + self.log_scale_var)
        
        return ctx, mean_per_point, logvar_per_point + self.log_scale_var, flag_per_point, loss_dict

@ENCODERS.register_module()
class EncoderWithKLLoss(PartEncoder):
    def __init__(self,  kl_weight=0.001,
                use_flow=False,
                latent_flow_depth=14,
                latent_flow_hidden_dim=256,
                **kwargs,):
        super().__init__(**kwargs)
        self.zdim = self.encoder.zdim
        self.kl_weight=kl_weight
        self.use_flow=use_flow 
        if use_flow:
            self.flow = build_latent_flow(latent_flow_depth, latent_flow_hidden_dim, self.encoder.zdim)
        self.zdim = self.encoder.zdim
    def get_part_code_params(self, input):
        return self.encoder(input)
    
    def sample_latents(self, sample_num, sample_points, device, fixed_id=None, valid_id=None, **kwargs):
        z = torch.randn(sample_num, self.zdim).to(device)
        if self.use_flow:
            z = self.flow(z, reverse=True).view(sample_num, self.zdim)
        
        part_code = z.unsqueeze(1).repeat_interleave(sample_points, dim=1)
        mean, logvar = self.get_params_from_part_code(part_code, valid_id)
        
        ids = torch.arange(self.n_class).to(device).unsqueeze(0) * valid_id
        seg_mask = ids.to(torch.int32).unsqueeze(-1).expand(-1, -1, sample_points // self.n_class).reshape(sample_num, -1)
        print(seg_mask.shape)
        mean_per_point, logvar_per_point, _ = self.gather_all(seg_mask, anchors=mean, variances=logvar)
        return [part_code], mean_per_point, logvar_per_point, seg_mask, valid_id

    def forward(self, pcds, device, **kwargs):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        ref = pcds['ref'].to(device).transpose(1,2)
        valid_id = pcds['present'].to(device)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        loss_dict = dict()
        B = ref.shape[0]
        mean, logvar = self.get_part_code_params(ref)
        mean, logvar = mean.reshape(B, self.zdim), logvar.reshape(B, self.zdim)
        z = reparameterize_gaussian(mean=mean, logvar=logvar).reshape(B, -1)  # (B, F)
        if self.use_flow:
            # P(z), Prior probability, parameterized by the flow: z -> w.
            w, delta_log_pw = self.flow(z.reshape(B, self.zdim), torch.zeros([B, 1]).to(z), reverse=False)
            log_pw = standard_normal_logprob(w).view(B, -1).sum(dim=1, keepdim=True)   # (B, 1)
            log_pz = log_pw - delta_log_pw.view(B, 1)  # (B, 1)
        else:
            log_pz = standard_normal_logprob(z).sum(dim=1)
        entropy = gaussian_entropy(logvar=logvar)
        loss_prior = (- log_pz - entropy).mean()
        loss_dict['prior_loss'] = self.kl_weight * loss_prior
        
        part_code = z.unsqueeze(-1).expand(-1, -1, self.n_class)
        mean, logvar = self.get_params_from_part_code(part_code, valid_id)
        
        mean_per_point, logvar_per_point, flag_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar, valid_id=valid_id)
        assert mean_per_point.shape == logvar_per_point.shape == ref.shape
        if self.training and self.part_aligner is not None:
            fit_loss = -gaussian_log_likelihood(ref, mean_per_point, logvar_per_point).mean(1, keepdim=True) * flag_per_point
            loss_dict['fit_loss'] = self.fit_loss_weight * fit_loss.sum(dim=(-1, -2)) / flag_per_point.sum(dim=(-1, -2))
        else:
            fit_loss = torch.zeros([1]).to(device)
        ctx = self.prepare_ctx( part_code, mean, logvar, anchor_assignments=seg_mask, z=z)
        return ctx, mean_per_point, logvar_per_point, flag_per_point, loss_dict
    
@ENCODERS.register_module()
class PartEncoderNoMixer(PartEncoder):
    def __init__(self, encoder, n_class, part_aligner=None, include_var=False, fit_loss_weight=1.0):
        super().__init__(encoder, None, n_class, part_aligner=part_aligner, include_var=include_var, fit_loss_weight=fit_loss_weight)
    
    def get_params_from_part_code(self, part_code, valid_id):
        B, C, N = part_code.shape
        if self.part_aligner is not None:
            mean, logvar = self.part_aligner(part_code, valid_id)
            l = [mean]
            if self.include_var:
                l.append(logvar)
            else:
                logvar = None
        else:
            mean = logvar = None
        return torch.zeros(B, C).to(part_code.device), part_code, mean, logvar
        
        
        
    