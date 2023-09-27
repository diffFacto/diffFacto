import torch 
import torch.nn as nn
import torch.nn.functional as F
from difffacto.utils.registry import ENCODERS
from difffacto.models.modules import ConditionalBatchNorm1d
from difffacto.utils.misc import timestep_embedding

@ENCODERS.register_module()
class SupSegsEncoder(nn.Module):
    def __init__(self, sup_segs_dim, part_latent_dim, num_anchors):
        super().__init__()
        dim = sup_segs_dim
        self.num_anchors = num_anchors
        self.sup_seg_enc = PartglotSupSegsEncoder(dim)

        self.first_conv = nn.Sequential(
            nn.Conv1d(dim, 128, 1),
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

    def forward(self, x, mask):
        neg_inf_mask = torch.logical_not(mask) * -10000
        B, n_seg, npoint, _ = x.shape
        x = x.reshape(B*n_seg, npoint, -1)
        out = self.sup_seg_enc(x) # (B * n_seg,  64, 1)
        feature_super_seg = torch.max(out, dim=1)[0].reshape(B, n_seg, -1).transpose(1,2)                       # (B, 64, n_seg)
        feature = self.first_conv(feature_super_seg)
        masked_feature = feature + neg_inf_mask.unsqueeze(1)
        feature_global = torch.max(masked_feature, dim=2, keepdim=True)[0]
        masked_feature = self.second_conv(torch.cat([feature, feature_global.expand(-1, -1, n_seg)], dim=1)) + neg_inf_mask.unsqueeze(1)
        feature_global = torch.max(masked_feature, dim=2, keepdim=False)[0]

        part_latent = self.latent_mlp(feature_global).reshape(B, self.num_anchors, -1)
        

        return part_latent

class PartglotSupSegsEncoder(nn.Module):
    def __init__(self, sup_segs_dim):
        super().__init__()
        dim = sup_segs_dim
        self.conv1 = nn.Sequential(nn.Conv1d(3, dim, 1), nn.BatchNorm1d(dim))
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.BatchNorm1d(dim))
        self.conv3 = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.BatchNorm1d(dim))
        self.conv4 = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.BatchNorm1d(dim))
        self.fc = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.conv4(out3))

        out = self.fc(out4)
        out = out.transpose(1, 2)

        return out

@ENCODERS.register_module()
class PartglotSupSegsEncoderWithCBN(nn.Module):
    def __init__(self, sup_segs_dim, embed_t_size=128, emb_size=256):
        super().__init__()
        dim = sup_segs_dim
        self.conv1 = nn.Conv1d(3, dim, 1)
        self.bn1=ConditionalBatchNorm1d(dim, embed_t_size, emb_size)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn2 = ConditionalBatchNorm1d(dim, embed_t_size, emb_size)
        self.conv3 = nn.Conv1d(dim, dim, 1)
        self.bn3 = ConditionalBatchNorm1d(dim, embed_t_size, emb_size)
        self.conv4 = nn.Conv1d(dim, dim, 1)
        self.bn4 = ConditionalBatchNorm1d(dim, embed_t_size, emb_size)
        self.fc = nn.Conv1d(dim, dim, 1)

    def forward(self, x, t):
        x = x.transpose(1, 2)
        out1 = F.relu(self.bn1(self.conv1(x)), t)
        out2 = F.relu(self.bn2(self.conv2(out1), t))
        out3 = F.relu(self.bn3(self.conv3(out2), t))
        out4 = F.relu(self.bn4(self.conv4(out3), t))

        out = self.fc(out4)
        out = out.transpose(1, 2)

        return out