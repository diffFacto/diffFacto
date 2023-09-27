import torch
import numpy as np
from torch import nn
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS
from difffacto.utils.misc import standard_normal_logprob, truncated_normal, gaussian_log_likelihood
from pointnet2_ops.pointnet2_utils import gather_operation
from .flow_utils.flow import get_point_cnf, get_latent_cnf




@MODELS.register_module()
class PointFlow(nn.Module):
    def __init__(self, 
                 encoder, 
                 input_dim, 
                 zdim, 
                 use_latent_flow, 
                 use_deterministic_encoder, 
                 point_cnf,
                 latent_cnf,
                 part_aligner,
                 prior_weight=1., 
                 recon_weight=1.,
                 entropy_weight=1.,

                 n_class=4
                 ):
        super(PointFlow, self).__init__()
        self.input_dim = input_dim
        self.zdim = zdim
        self.use_latent_flow = use_latent_flow
        self.use_deterministic_encoder = use_deterministic_encoder
        self.prior_weight = prior_weight
        self.recon_weight = recon_weight
        self.entropy_weight = entropy_weight
        self.truncate_std = None
        self.encoder = build_from_cfg(encoder, ENCODERS)
        self.part_aligner=build_from_cfg(part_aligner, ENCODERS)
        self.point_cnf = get_point_cnf(point_cnf)
        self.latent_cnf = [get_latent_cnf(latent_cnf) for _ in range(n_class)] if use_latent_flow else nn.Sequential()
        self.n_class=n_class
        
    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float().cuda()
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)
    
    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def gather_all(self, anchor_assignments, anchors=None, variances=None):
        B, N = anchor_assignments.shape
        anchor_per_point = torch.zeros(B, 3, N).to(anchor_assignments.device) 
        variance_per_point = torch.zeros(B, 3, N).to(anchor_assignments.device) 
        if anchors is not None:
            anchor_per_point = gather_operation(anchors.contiguous(), anchor_assignments).reshape(B, 3, N)  
        if variances is not None:
            variance_per_point = gather_operation(variances.contiguous(), anchor_assignments).reshape(B, 3, N) 
        
        return anchor_per_point, variance_per_point
    
    def forward(self, pcds, device='cuda', **kwargs):
        input = pcds['input'].to(device)
        ref = pcds['ref'].to(device)
        seg_mask = pcds['ref_seg_mask'].to(device).to(torch.int32)
        seg_flag = pcds['attn_map'].to(device)
        valid_id = pcds.get('present', None)
        if valid_id is not None:
            valid_id = valid_id.to(device)
        B, N, C = ref.shape
        if not self.training:
            _, pred = self.reconstruct(torch.cat([input, seg_flag], dim=-1), N, None, valid_id) if self.use_deterministic_encoder else self.sample(B, N, None, None, device, valid_id)
            return [(dict(pred=pred, input_ref=ref, shift=pcds['shift'], scale=pcds['scale']), "gen" if not self.use_deterministic_encoder else "sample")]
        z_mu, z_sigma = self.encoder(torch.cat([input, seg_flag], dim=-1))
        assert z_mu.shape[1] == z_sigma.shape[1] == self.n_class
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)
        mean, logvar = self.part_aligner(z.transpose(1,2), mask=valid_id)
        mean_per_point, logvar_per_point = self.gather_all(seg_mask, anchors=mean, variances=logvar)
        assert mean_per_point.shape == logvar_per_point.shape == ref.transpose(1,2).shape
        if self.training and self.part_aligner is not None:
            fit_loss = -gaussian_log_likelihood(ref.transpose(1,2), mean_per_point, logvar_per_point).mean()
        # Compute H[Q(z|X)]
        if self.use_deterministic_encoder:
            entropy = torch.zeros(B).to(z)
        else:
            entropy = self.gaussian_entropy(z_sigma.reshape(B* self.n_class, -1))

        # Compute the prior probability P(z)
        if self.use_latent_flow:
            log_pz = torch.zeros(B, self.n_class, 1).to(input)
            for i in range(self.n_class):
                w, delta_log_pw = self.latent_cnf[i](z[:, i], None, torch.zeros(B, 1).to(z))
                log_pw = standard_normal_logprob(w).view(B, -1).sum(1, keepdim=True)
                delta_log_pw = delta_log_pw.view(B, 1)
                log_pz[:, i] = log_pw - delta_log_pw * valid_id[:, i].unsqueeze(1)
            log_pz = log_pz.sum(1, keepdim=True) / valid_id.sum(1, keepdim=True)
        else:
            log_pz = torch.zeros(B, 1).to(z)

        # Compute the reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        z_new = torch.cat([z_new, mean.transpose(1,2), torch.exp(logvar).transpose(1,2)], dim=-1)
        z_new = z_new.reshape(B, -1)
        y, delta_log_py = self.point_cnf(ref, z_new, torch.zeros(B, N, 1).to(ref))
        log_py = standard_normal_logprob(y).view(B, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(B, N, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight

        return {
            'entropy_loss':entropy_loss,
            'recon_loss':recon_loss,
            'prior_loss':prior_loss,
            'fit_loss': fit_loss
        }

    def decode(self, z, num_points, truncate_std=None, valid_id=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        mean, logvar = self.part_aligner(z.transpose(1,2), mask=valid_id)
        z = torch.cat([z, mean.transpose(1,2), torch.exp(logvar).transpose(1,2)], dim=-1)
        x = self.point_cnf(y, z.reshape(z.size(0), -1), reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None, valid_id=None):
        assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
        # Generate the shape code from the prior
        z = torch.zeros([batch_size, self.n_class, self.zdim]).to(gpu)
        for i in range(self.n_class):
            w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
            z[:, i] = self.latent_cnf[i](w, None, reverse=True).view(*w.size())
        # Sample points conditioned on the shape code
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        mean, logvar = self.part_aligner(z.transpose(1,2), mask=valid_id)
        z_new = torch.cat([z, mean.transpose(1,2), torch.exp(logvar).transpose(1,2)], dim=-1)
        
        x = self.point_cnf(y, z_new.reshape(batch_size, -1), reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None, valid_id=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        y, x = self.decode(z, num_points, truncate_std, valid_id=valid_id)
        return y, x
