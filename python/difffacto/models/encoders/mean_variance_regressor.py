import torch
import torch.nn.functional as F
from torch import nn
from difffacto.utils.registry import ENCODERS

@ENCODERS.register_module()
class MeanVarianceRegressor(nn.Module):
    def __init__(self, latent_dim, num_class):
        super().__init__()
        self.num_class = num_class
        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(latent_dim, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, 256)
        self.fc4_m = nn.Linear(256, latent_dim)
        self.fc_bn1_m = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(num_class)]) 
        self.fc_bn2_m = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(num_class)]) 
        self.fc_bn3_m = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(num_class)]) 
        

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(latent_dim, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, 256)
        self.fc4_v = nn.Linear(256, latent_dim)
        self.fc_bn1_v = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(num_class)]) 
        self.fc_bn2_v = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(num_class)]) 
        self.fc_bn3_v = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(num_class)]) 
        

    def forward(self, x):
        m = F.relu(torch.stack([self.fc_bn1_m[i](self.fc1_m(x)[:, i]) for i in range(self.num_class)], dim=1))
        m = F.relu(torch.stack([self.fc_bn2_m[i](self.fc2_m(m)[:, i]) for i in range(self.num_class)], dim=1))
        m = F.relu(torch.stack([self.fc_bn3_m[i](self.fc3_m(m)[:, i]) for i in range(self.num_class)], dim=1))
        m = self.fc4_m(m)
        v = F.relu(torch.stack([self.fc_bn1_v[i](self.fc1_v(x)[:, i]) for i in range(self.num_class)], dim=1))
        v = F.relu(torch.stack([self.fc_bn2_v[i](self.fc2_v(v)[:, i]) for i in range(self.num_class)], dim=1))
        v = F.relu(torch.stack([self.fc_bn3_v[i](self.fc3_v(v)[:, i]) for i in range(self.num_class)], dim=1))
        v = self.fc4_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v