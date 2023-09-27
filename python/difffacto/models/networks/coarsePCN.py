import torch
import torch.nn as nn
from difffacto.utils.registry import MODELS, METRICS, build_from_cfg

@MODELS.register_module()
class CoarsePCN(nn.Module):
    def __init__(self, num_anchors, encoder_channel, loss):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.number_coarse = num_anchors
        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.number_coarse)
        )
        self.loss_func = build_from_cfg(loss, METRICS)

    def forward(self, pcds, device):
        gt = pcds['pointcloud'].to(device)
        xyz = pcds['pointcloud'].to(device)
        bs , n , _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        # decoder
        coarse = self.mlp(feature_global).reshape(-1,self.number_coarse,3) # B M 3
        
        
        if self.training:
            loss_coarse = self.loss_func(coarse, gt)
            return dict(loss_coarse=loss_coarse)
        return dict(pred=coarse.contiguous(), ref=gt, input=xyz)
