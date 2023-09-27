import torch
import torch.nn as nn
import torch.nn.functional as F
from difffacto.utils.registry import GENERATORS


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

@GENERATORS.register_module()
class Generator(nn.Module):

    def __init__(self, inp_dim, out_dim, use_bn, output_bn, dims, prior='gaussian', gaussian_scale=1.0):
        super().__init__()


        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.output_bn = output_bn
        self.dims = dims
        self.gaussian_scale=gaussian_scale
        curr_dim = self.inp_dim
        self.layers = []
        self.bns = []
        for hid in self.dims:
            self.layers.append(nn.Linear(curr_dim, hid))
            self.bns.append(nn.BatchNorm1d(hid))
            curr_dim = hid
        self.layers = nn.ModuleList(self.layers)
        self.bns = nn.ModuleList(self.bns)
        self.out = nn.Linear(curr_dim, self.out_dim)
        self.out_bn = nn.BatchNorm1d(self.out_dim)
        self.prior_type = prior

    def get_prior(self, bs):
        if self.prior_type == "truncate_gaussian":
            noise = (torch.randn(bs, self.inp_dim) * self.gaussian_scale).cuda()
            noise = truncated_normal(
                noise, mean=0, std=self.gaussian_scale, trunc_std=2.)
            return noise
        elif self.prior_type == "gaussian":
            return torch.randn(bs, self.inp_dim).cuda() * self.gaussian_scale
        else:
            raise NotImplementedError(
                "Invalid prior type:%s" % self.prior_type)

    def forward(self, z=None, bs=None):
        if z is None:
            assert bs is not None
            z = self.get_prior(bs).cuda()

        y = z
        for layer, bn in zip(self.layers, self.bns):
            y = layer(y)
            if self.use_bn:
                y = bn(y)
            y = F.relu(y)
        y = self.out(y)

        if self.output_bn:
            y = self.out_bn(y)
        return y

