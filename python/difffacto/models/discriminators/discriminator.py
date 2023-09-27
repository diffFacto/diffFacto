import torch
import torch.nn as nn
import torch.nn.functional as F
from difffacto.utils.registry import DISCRIMINATORS

@DISCRIMINATORS.register_module()
class Discriminator(nn.Module):

    def __init__(self, inp_dim, use_bn, use_ln, use_sigmoid, dims):
        super().__init__()

        self.inp_dim = inp_dim
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_sigmoid = use_sigmoid
        self.dims = dims

        curr_dim = self.inp_dim
        self.layers = []
        self.bns = []
        self.lns = []
        for hid in self.dims:
            self.layers.append(nn.Linear(curr_dim, hid))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(hid))
            else:
                self.bns.append(None)
            if self.use_ln:
                self.lns.append(nn.LayerNorm(hid))
            else:
                self.lns.append(None)
            curr_dim = hid
        self.layers = nn.ModuleList(self.layers)
        self.bns = nn.ModuleList(self.bns)
        self.lns = nn.ModuleList(self.lns)
        self.out = nn.Linear(curr_dim, 1)

    def forward(self, z=None, bs=None, return_all=False):
        if z is None:
            assert bs is not None
            z = torch.randn(bs, self.inp_dim).cuda()

        y = z
        for layer, bn, ln in zip(self.layers, self.bns, self.lns):
            y = layer(y)
            if self.use_bn:
                y = bn(y)
            if self.use_ln:
                y = ln(y)
            y = F.leaky_relu(y, 0.2)
        y = self.out(y)

        if self.use_sigmoid:
            y = torch.sigmoid(y)
        if return_all:
            return {
                'x': y
            }
        else:
            return y


@DISCRIMINATORS.register_module()
class DiscriminatorcGan(nn.Module):

    def __init__(self, inp_dim, ctx_dim, inner_dim, n_class):
        super().__init__()

        self.proj_in = nn.Conv1d(inp_dim, inner_dim, 1)
        self.n_class=n_class
        self.seq = nn.Sequential(
            nn.Linear(inner_dim*n_class + ctx_dim*n_class, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, z=None, ctx=None, mask=None):
        bs = z.shape[0]
        z = self.proj_in(z)
        if mask is not None:
            z = z * mask.unsqueeze(1)
            ctx = ctx * mask.unsqueeze(1)
        inp = torch.cat([z, ctx], dim=1).transpose(1,2).reshape(bs, -1)
        y = self.seq(inp)

        return y
