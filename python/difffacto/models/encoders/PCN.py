import torch
import torch.nn as nn
import torch.nn.functional as F
from difffacto.utils.registry import ENCODERS

@ENCODERS.register_module()
class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)
    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(
        self, 
        point_dim=3, 
        part_latent_dim=512, 
        num_anchors=4,
        normalize_latent=False
        ):
        super().__init__()

        self.point_dim=point_dim
        self.normalize_latent=normalize_latent
        self.num_anchors=num_anchors
        self.part_latent_dim=part_latent_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(point_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )

        self.latent_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, part_latent_dim * num_anchors)
        )

    def forward(self, xyz=None):
        B, N, C = xyz.shape
        assert C == self.point_dim
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        part_latent = self.latent_mlp(feature_global).reshape(B, self.num_anchors, -1)
        if self.normalize_latent:
            part_latent = F.normalize(part_latent.reshape(B*self.num_anchors, -1)).reshape(B, self.num_anchors, -1)
        return part_latent
