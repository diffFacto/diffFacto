import torch 
from torch import nn
import torch.nn.functional as F
from difffacto.utils.registry import ENCODERS

@ENCODERS.register_module()
class PointNetVAEBase(nn.Module):
    def __init__(self, point_dim=3,  zdim=1024, **kwargs):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(point_dim, 128 , 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)
        
        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x, mask=None):
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        if mask is not None:
            INF = float('inf')
            _mask = torch.zeros_like(x)
            _mask.masked_fill_(mask[:, None], -1.0 * float('inf'))
            x = x + _mask
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        x = torch.nan_to_num(x, neginf=0.0)
        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v


@ENCODERS.register_module()
class PointNetVAE(nn.Module):
    def __init__(self, point_dim=7,  zdim=1024, num_anchors=4):
        super().__init__()
        self.zdim = zdim
        self.num_anchors=num_anchors
        self.conv1 = nn.Conv1d(point_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim*num_anchors)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)
        
        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim*num_anchors)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, -1)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m.reshape(B, self.num_anchors, self.zdim), v.reshape(B, self.num_anchors, self.zdim)

@ENCODERS.register_module()
class PointNet(PointNetVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        B = x.shape[0]
        m, v = super().forward(x)
        return m.reshape(B, self.num_anchors, self.zdim)
    

    
@ENCODERS.register_module()
class PointNetV2(nn.Module):
    def __init__(self, point_dim=3, zdim=1024, num_anchors=4, reweight_by_anchor=True, use_ln=False, per_part_mlp=False):
        super().__init__()
        self.reweight_by_anchor=reweight_by_anchor
        self.per_part_mlp=per_part_mlp
        self.zdim = zdim
        self.num_anchors=num_anchors
        self.use_ln=use_ln
        self.conv1 = nn.Conv1d(point_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if use_ln:
            self.mlp_m = nn.Sequential(nn.Linear(512, 256), 
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128), 
                                     nn.LayerNorm(128),
                                     nn.ReLU(),
                                     nn.Linear(128, zdim))
            self.mlp_v = nn.Sequential(nn.Linear(512, 256), 
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128), 
                                     nn.LayerNorm(128),
                                     nn.ReLU(),
                                     nn.Linear(128, zdim))
        else:
            self.mlp_m = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Conv1d(256, 128, 1), 
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Conv1d(128, zdim, 1))
            self.mlp_v = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Conv1d(256, 128, 1), 
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Conv1d(128, zdim, 1))
        if self.per_part_mlp:
            self.mlp_m = nn.Sequential(nn.Conv1d(512*self.num_anchors, 256*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(256*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(256*self.num_anchors, 128*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(128*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(128*self.num_anchors, zdim*self.num_anchors, 1, groups=self.num_anchors))
            self.mlp_v = nn.Sequential(nn.Conv1d(512*self.num_anchors, 256*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(256*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(256*self.num_anchors, 128*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(128*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(128*self.num_anchors, zdim*self.num_anchors, 1, groups=self.num_anchors))

    def forward(self, x, attn_weight):
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        weighted_x = x.unsqueeze(-1) * attn_weight.unsqueeze(1)
        if self.reweight_by_anchor:
            weighted_x = weighted_x * self.num_anchors
        x = torch.max(weighted_x, 2, keepdim=True)[0]
        x = x.view(B, 512, self.num_anchors)
        
        if self.per_part_mlp:
            x = x.transpose(1,2).reshape(B, -1, 1)
            m = self.mlp_m(x).reshape(B, self.num_anchors, -1)
            v = self.mlp_v(x).reshape(B, self.num_anchors, -1)
        else:
            if self.use_ln:
                x = x.transpose(1,2)
                m = self.mlp_m(x)
                v = self.mlp_v(x)
            else:
                m = self.mlp_m(x).transpose(1,2)
                v = self.mlp_v(x).transpose(1,2)
        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v
    
@ENCODERS.register_module()
class PointNetV2Variant(nn.Module):
    def __init__(self, point_dim=3, zdim=1024, num_anchors=4, per_part_mlp=False):
        super().__init__()
        self.per_part_mlp=per_part_mlp
        self.zdim = zdim
        self.num_anchors=num_anchors
        self.pnet = nn.Sequential(nn.Conv1d(point_dim, 128, 1), 
                                  nn.BatchNorm1d(128), 
                                  nn.ReLU(),
                                  nn.Conv1d(128, 128, 1),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Conv1d(128, 256, 1),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Conv1d(256, 512, 1),
                                  nn.BatchNorm1d(512))

        self.mlp_m = nn.ModuleList([nn.Sequential(nn.Conv1d(512, 256, 1), 
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Conv1d(256, 128, 1), 
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Conv1d(128, zdim, 1)) for _ in range(self.num_anchors)])
        self.mlp_v = nn.ModuleList([nn.Sequential(nn.Conv1d(512, 256, 1), 
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Conv1d(256, 128, 1), 
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Conv1d(128, zdim, 1)) for _ in range(self.num_anchors)])
        if not self.per_part_mlp:
            self.mlp_m = self.mlp_m[0]
            self.mlp_v = self.mlp_v[0]
    def get_part_parameters(self, part_id):
        assert 0 <= part_id < self.num_anchors
        return list(self.pnet.parameters) + list(self.mlp_m[part_id].parameters()) + list(self.mlp_v[part_id].parameters())
    def forward(self, x, seg_flag, part_id=-1):
        assert x.shape[-1] == 3 + self.num_anchors
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = self.pnet(x)
        weighted_x = x.unsqueeze(-1) * seg_flag.unsqueeze(1) - (1 - seg_flag.unsqueeze(1)) * float('inf')
        x = torch.max(weighted_x, 2, keepdim=True)[0]
        x = torch.nan_to_num(x, 0, 0, 0)
        x = x.view(B, 512, self.num_anchors)
        if part_id >= 0:
            return self.mlp_m[part_id](x[..., part_id]), self.mlp_v[part_id](x[..., part_id])
        if self.per_part_mlp:
            m, v = [], []
            for i in range(self.num_anchors):
                mm = self.mlp_m[i](x[..., i])
                vv = self.mlp_v[i](x[..., i])
                m.append(mm)
                v.append(vv)
            m = torch.stack(m, dim=1)
            v = torch.stack(v, dim=1)
        else:
            m = self.mlp_m(x).transpose(1,2)
            v = self.mlp_v(x).transpose(1,2)
        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v

@ENCODERS.register_module()
class PointNetV2VAE(nn.Module):
    def __init__(self, point_dim=3, zdim=1024, num_anchors=4, reweight_by_anchor=True, use_ln=False, per_part_mlp=False, deterministic=False):
        super().__init__()
        self.reweight_by_anchor=reweight_by_anchor
        self.per_part_mlp=per_part_mlp
        self.zdim = zdim
        self.num_anchors=num_anchors
        self.use_ln=use_ln
        self.conv1 = nn.Conv1d(point_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.deterministic=deterministic

        if use_ln:
            self.mlp_m = nn.Sequential(nn.Linear(512, 256), 
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128), 
                                     nn.LayerNorm(128),
                                     nn.ReLU(),
                                     nn.Linear(128, zdim))
            self.mlp_v = nn.Sequential(nn.Linear(512, 256), 
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128), 
                                     nn.LayerNorm(128),
                                     nn.ReLU(),
                                     nn.Linear(128, zdim))
        else:
            self.mlp_m = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Conv1d(256, 128, 1), 
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Conv1d(128, zdim, 1))
            self.mlp_v = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Conv1d(256, 128, 1), 
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Conv1d(128, zdim, 1))
        if self.per_part_mlp:
            self.mlp_m = nn.Sequential(nn.Conv1d(512*self.num_anchors, 256*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(256*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(256*self.num_anchors, 128*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(128*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(128*self.num_anchors, zdim*self.num_anchors, 1, groups=self.num_anchors))
            self.mlp_v = nn.Sequential(nn.Conv1d(512*self.num_anchors, 256*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(256*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(256*self.num_anchors, 128*self.num_anchors, 1, groups=self.num_anchors), 
                                     nn.BatchNorm1d(128*self.num_anchors),
                                     nn.ReLU(),
                                     nn.Conv1d(128*self.num_anchors, zdim*self.num_anchors, 1, groups=self.num_anchors))

    def forward(self, x):
        assert x.shape[-1] == 3 + self.num_anchors
        attn_weight = x[..., 3:]
        x = x[..., :3]
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        weighted_x = x.unsqueeze(-1) * attn_weight.unsqueeze(1)
        if self.reweight_by_anchor:
            weighted_x = weighted_x * self.num_anchors
        x = torch.max(weighted_x, 2, keepdim=True)[0]
        x = x.view(B, 512, self.num_anchors)
        
        if self.per_part_mlp:
            x = x.transpose(1,2).reshape(B, -1, 1)
            m = self.mlp_m(x).reshape(B, self.num_anchors, -1)
            v = self.mlp_v(x).reshape(B, self.num_anchors, -1)
        else:
            if self.use_ln:
                x = x.transpose(1,2)
                m = self.mlp_m(x)
                v = self.mlp_v(x)
            else:
                m = self.mlp_m(x).transpose(1,2)
                v = self.mlp_v(x).transpose(1,2)
        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        if self.deterministic:
            return m
        return m, v


@ENCODERS.register_module()
class PointNetV3(nn.Module):
    def __init__(self, point_dim=3, zdim=1024, num_anchors=4, mult=1, reweight_by_anchor=True):
        super().__init__()
        self.mult = mult
        self.reweight_by_anchor=reweight_by_anchor
        self.zdim = zdim
        self.num_anchors=num_anchors
        self.conv1 = nn.Conv1d(point_dim, 128 * mult, 1)
        self.conv2 = nn.Conv1d(128* mult, 128* mult, 1)
        self.conv3 = nn.Conv1d(128* mult, 256* mult, 1)
        self.conv4 = nn.Conv1d(256* mult, 256* mult, 1)
        self.bn1 = nn.BatchNorm1d(128* mult)
        self.bn2 = nn.BatchNorm1d(128* mult)
        self.bn3 = nn.BatchNorm1d(256* mult)
        self.bn4 = nn.BatchNorm1d(256* mult)

        # Mapping to [c], cmean
        self.fc1_m = nn.Conv1d(512* mult, 256* mult, 1)
        self.fc2_m = nn.Conv1d(256* mult, 128* mult, 1)
        self.fc3_m = nn.Conv1d(128* mult, zdim, 1)
        self.fc_bn1_m = nn.BatchNorm1d(256* mult)
        self.fc_bn2_m = nn.BatchNorm1d(128* mult)

    def forward(self, x):
        if x.shape[-1] == 3 + self.num_anchors:
            attn_weight = x[..., 3:]
            x = x[..., :3]
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        weighted_x = x.unsqueeze(-1) * attn_weight.unsqueeze(1)
        if self.reweight_by_anchor:
            weighted_x = weighted_x * self.num_anchors
        weighted_x = torch.max(weighted_x, 2, keepdim=True)[0]
        x = torch.max(x, 2, keepdim=True)[0].unsqueeze(-1).expand(-1, -1, -1, self.num_anchors)
        x = torch.cat([x, weighted_x], dim=1).view(B, 512, self.num_anchors)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m.transpose(1,2)