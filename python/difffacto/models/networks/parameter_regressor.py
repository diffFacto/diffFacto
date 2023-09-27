import torch
import numpy as np
from torch import nn
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS
from difffacto.utils.misc import standard_normal_logprob, truncated_normal, gaussian_log_likelihood
from pointnet2_ops.pointnet2_utils import gather_operation
from .flow_utils.flow import get_point_cnf, get_latent_cnf




@MODELS.register_module()
class Aligner(nn.Module):
    def __init__(self, 
                 encoder,
                n_class
                 ):
        super(Aligner, self).__init__()
        self.encoder = build_from_cfg(encoder, ENCODERS)
        self.regressor = nn.Sequential(nn.Linear(self.encoder.zdim, 512), nn.ReLU(), 
                                       nn.Linear(512, 512), nn.ReLU(),
                                       nn.Linear(512, 512), nn.ReLU(),
                                       nn.Linear(512, 512), nn.ReLU(),
                                       nn.Linear(512, 64), nn.ReLU(),
                                       nn.Linear(64, 6))
        self.n_class=n_class
    def forward(self, pcds, device='cuda', **kwargs):
        ref = pcds['ref'].to(device)
        seg_mask = pcds['ref_seg_mask'].to(device)
        seg_flag = pcds['ref_attn_map'].to(device)
        valid_id = pcds.get('present', None)
        if valid_id is not None:
            valid_id = valid_id.to(device)

        loss_dict = dict()
        B, N, C = ref.shape

        codes = self.encoder(torch.cat([ref, seg_flag], dim=-1)).reshape(B, self.n_class, -1)
        params = self.regressor(codes)
        param_per_point = gather_operation(torch.cat([params.transpose(1,2), valid_id.unsqueeze(1)], dim=1).contiguous(), seg_mask.to(int)).reshape(B, 6+1, N)
        mean, logvar, flag = torch.split(param_per_point, [3, 3, 1], dim=1)
        fit_loss = - gaussian_log_likelihood(ref, mean, logvar).mean(1, keepdim=True) * flag
        fit_loss = fit_loss.sum(dim=(-1, -2)) / flag.sum(dim=(-1, -2))
        return dict(loss=fit_loss) if self.training else [(dict(index=pcds['id'] , mean=params[..., :3], logvar=params[..., 3:], loss=fit_loss), "params")]

