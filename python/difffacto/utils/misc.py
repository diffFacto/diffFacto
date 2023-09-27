import os 
import warnings
import torch
import numpy as np
import torch.nn.functional as F
import random
from pointnet2_ops import pointnet2_utils
import time
import glob
import math

"""
Adapted from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L124
"""

from typing import Callable, Iterable, Sequence, Union


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def str_list(argstr):
    return list(argstr.split(','))

def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def pad_unmasked_element_by_zero(tensor, mask):
    # tensor[B, n_seg]
    # mask [B, n_seg]
    B, n_seg = tensor.shape
    neg_mask = not mask.to(bool)
    zeroed_tensor = tensor.masked_fill(neg_mask, 0.)
    return zeroed_tensor
    

def get_index(coarse, fine):
    num_coarse = coarse.shape[1]
    B, num_fine = fine.shape[:2]
    num_per_anchor = num_fine // num_coarse
    dist = torch.sum(torch.pow(fine.reshape(B, num_coarse, num_per_anchor, 3).unsqueeze(2) - coarse.unsqueeze(2).unsqueeze(1), 2), -1) #[B, num_coarse, num_coarse, num_per_anchor]
    idx = torch.argmin(dist, dim=2).to(torch.int32)
    return idx.reshape(B, -1)


def parse_losses(losses):
    _losses = dict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            _losses[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            _losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    total_loss = sum(_value for _key, _value in _losses.items() if 'loss' in _key)
    return total_loss, _losses

def current_time():
    return time.asctime( time.localtime(time.time()))

def filter_points(data, npoints):
    n = data.shape[0]
    if n <= npoints:
        return data 
    else:
        idx = np.random.RandomState(2020).permutation(n)[:npoints]
        return data[idx]

def search_ckpt(work_dir):
    files = glob.glob(os.path.join(work_dir,"checkpoints/ckpt_*.pth"))
    if len(files)==0:
        return None
    files = sorted(files,key=lambda x:int(x.split("_")[-1].split(".pth")[0]))
    return files[-1]  

def check_interval(step,step_interval):
    if step is None or step_interval is None:
        return False 
    if step % step_interval==0:
        return True 
    return False 


def sync(data, to_numpy=True):
    """
        sync data and convert data to numpy
    """
    def _sync(data):
        if isinstance(data,(list,tuple)):
            data =  [_sync(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_sync(d) for k,d in data.items()}
        elif isinstance(data,torch.Tensor):
            data = data.detach().cpu()
            if to_numpy:
                data = data.numpy()
        elif not isinstance(data,(int,float,str,np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    return _sync(data) 

def build_file(work_dir,prefix):
    """ build file and makedirs the file parent path """
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix)>0:
        work_dir = os.path.join(work_dir,prefix)
    os.makedirs(work_dir,exist_ok=True)
    file = os.path.join(work_dir,file_name)
    return file 

def assign_anchor(data, anchors):
    dist = torch.sum(torch.pow(data.unsqueeze(2) - anchors.unsqueeze(1), 2), dim=-1) # B, N, M
    anchor_assignments = dist.argmin(dim=-1).to(data.device)
    return anchor_assignments

def fps(data, number, ret_id=False):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    if ret_id:
        return fps_data, fps_idx
    return fps_data

def separate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = np.random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).to(xyz.device)
        else:
            if isinstance(fixed_points,list):
                fixed_point = np.random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

def check_file(file,ext=None):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    if ext:
        if not os.path.splitext(file)[1] in ext:
            # warnings.warn(f"the type of {file} must be in {ext}")
            return False
    return True

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps

def gaussian_logprob(z, mean, logvar=None ):
    log_z = -0.5 * np.log(2 * np.pi)
    std_squared = torch.exp(logvar)
    return -logvar + log_z - (z - mean).pow(2) / (2. * std_squared)

def gaussian_entropy(logvar, dim=1):
    const = 0.5 * float(logvar.size(dim)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=dim, keepdim=False) + const
    return ent

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2
def gaussian_log_likelihood(z, mean=None, logvar=None, dim=3):
    """
    Compute the log probability of x belonging to the gaussian with mean and log variance
    
    param x: [B, 3, N] tensor
    param mean: [B, 3, N] tensor 
    param log_var: [B, 3, N] tensor 
    """
    log_z = -0.5 * np.log(2 * np.pi) * dim
    if logvar is None:
        std_squared = 1.
        logvar = 0.
    else:
        std_squared = torch.exp(logvar) if isinstance(logvar, torch.Tensor) else math.exp(logvar)
    if mean is None:
        mean = 0.
    return -logvar + log_z - (z - mean).pow(2) / (2. * std_squared)
    
    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
"""
def get_part_loss(anchors, gt_pcds, loss_func):
    loss = 0.
    actual_num=0
    num_classes=anchors.shape[1]
    for i in range(num_classes):
        batch_idx = part_means[:, i, 0] == i 
        if batch_idx.any():
            actual_num += 1
            batch_means = part_means[batch_idx, i, 1:]
            anchor = anchors[batch_idx, i]
            loss += loss_func(anchor, batch_means)

    return loss / actual_num

"""