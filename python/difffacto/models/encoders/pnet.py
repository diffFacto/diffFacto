import torch
import torch.nn as nn
from difffacto.utils.registry import ENCODERS

@ENCODERS.register_module()
class Pnet2Stage(nn.Module):
    def __init__(self, latent_dim=1024, point_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
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
            nn.Conv1d(512, self.latent_dim, 1)
        )

    def forward(self, xyz):
        B, N, C = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]
        
        return feature_global