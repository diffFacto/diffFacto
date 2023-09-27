import numpy as np
import torch


def normal_kl(mean1, logvar1, mean2, logvar2, dim=3):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """


    return 0.5 * (
        -dim
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    credit: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produces the cumulative product of (1-beta) up to that
                    part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                    prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = from_numpy_to_device(arr, timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def from_numpy_to_device(arr, device):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=device).float()
    return res

def get_cov_from_params(eigens, thetas=None):
    # thetas: [B, 3], eigens: [B, 3]
    B = eigens.shape[0]
    if thetas is None:
        thetas = torch.zeros_like(eigens)
    assert B == thetas.shape[0]
    tensor_0 = torch.zeros(B).to(eigens.device)
    tensor_1 = torch.ones(B).to(eigens.device)
    cos_thetas = torch.cos(thetas)
    sin_thetas = torch.sin(thetas)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0], dim=1),
                    torch.stack([tensor_0, cos_thetas[:, 0], -sin_thetas[:, 0]], dim=1),
                    torch.stack([tensor_0, sin_thetas[:, 0], cos_thetas[:, 0]], dim=1)], dim=1).reshape(B,3,3)

    RY = torch.stack([
                    torch.stack([cos_thetas[:, 1], tensor_0, sin_thetas[:, 1]], dim=1),
                    torch.stack([tensor_0, tensor_1, tensor_0], dim=1),
                    torch.stack([-sin_thetas[:, 1], tensor_0, cos_thetas[:, 1]], dim=1)], dim=1).reshape(B, 3,3)

    RZ = torch.stack([
                    torch.stack([cos_thetas[:,2], -sin_thetas[:,2], tensor_0], dim=1),
                    torch.stack([sin_thetas[:,2], cos_thetas[:,2], tensor_0], dim=1),
                    torch.stack([tensor_0, tensor_0, tensor_1], dim=1)], dim=1).reshape(B,3,3)

    R = torch.bmm(RZ, RY)
    R = torch.bmm(R, RX)
    cov = torch.bmm(R, eigens.reshape(B, 3,1) * R.transpose(1,2))
    sqrt_cov = R * torch.sqrt(eigens.reshape(B,1,3))
    return cov, sqrt_cov