import torch 
from torch.nn import Module
from difffacto.utils.registry import DIFFUSIONS, NETS, build_from_cfg
from difffacto.utils.constants import ModelMeanType, ModelVarType, LossType, model_mean_dict, model_var_dict
from difffacto.utils.misc import *
from difffacto.utils.constants import *
import math
import numpy as np
from .diffusion_utils import extract_into_tensor, betas_for_alpha_bar, normal_kl






@DIFFUSIONS.register_module()
class PointDiffusion(Module):

    def __init__(
        self, 
        net, 
        num_timesteps, 
        beta_1, 
        beta_T, 
        mode='linear', 
        use_beta=True, 
        rescale_timesteps=False, 
        loss_type='mse', 
        model_mean_type='epsilon', 
        model_var_type='fixed_small', 
        include_global_latent=False, 
        include_anchor_latent=True,
        scaled_loss=False,
        include_anchors=False
    ):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.model = build_from_cfg(net, NETS)
        self.model_mean_type = model_mean_dict[model_mean_type]
        self.include_global_latent=include_global_latent
        self.include_anchor_latent=include_anchor_latent
        self.include_anchors=include_anchors
        self.model_var_type = model_var_dict[model_var_type]
        self.loss_type = model_loss_dict[loss_type]
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.scaled_loss=scaled_loss
        self.use_beta = use_beta
        self.rescale_timesteps = rescale_timesteps
        if mode == 'linear':
            betas = np.linspace(beta_1, beta_T, num=num_timesteps, dtype=np.float64)
        elif mode == 'cosine':
            betas = betas_for_alpha_bar(num_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
            
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(num_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )        
        
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_0: the [N x 3 x n] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param x_0: the [N x 3 x n] tensor of anchors the tensor of shape as x_0 to which x_0's diffuses
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, x_start.shape
        )
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_0: the initial data batch.
        :param anchors: anchors for each point in x_0
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_0.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
        
    def q_posterior_mean(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t 
        )
        assert (
            posterior_mean.shape[0]
            == x_t.shape[0]
        )
        return posterior_mean
    
    def q_posterior_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_t.shape[0]
        )
        return posterior_variance, posterior_log_variance_clipped
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
    
        
        posterior_mean = self.q_posterior_mean(x_start, x_t, t)
        posterior_variance, posterior_log_variance_clipped = self.q_posterior_variance(x_start, x_t, t)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    
    def p_mean_variance(
        self, x, t, anchors=None, code=None, pointwise_latent=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        B, C = x.shape[:2]
        assert t.shape == (B,)
        if self.include_anchors:
            input = torch.cat([x, anchors], dim=1)
        else:
            input = x
        if self.include_anchor_latent:
            assert pointwise_latent is not None 
            ctx = pointwise_latent
        elif self.include_global_latent:
            assert code is not None
            ctx = code 
        else:
            ctx = None
        model_output = self.model(input, self._scale_timesteps(t), ctx)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
        
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, eps.shape) * eps
                )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t - coef3 * anchors) / coef1
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            ) * x_t
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        if self.use_beta:
            return extract_into_tensor(self.betas, t, t.shape)
        return t

    def p_sample(
        self, x, t, anchors=None, code=None,pointwise_latent=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            x,
            t,
            anchors=anchors,
            code=code,
            pointwise_latent=pointwise_latent
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def p_sample_loop(
        self,
        shape,
        anchors=None,
        code=None,
        pointwise_latent=None,
        noise=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for t, sample in self.p_sample_loop_progressive(
            shape,
            noise=noise,
            anchors=anchors,
            code=code,
            pointwise_latent=pointwise_latent,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        shape,
        anchors=None,
        code=None,
        pointwise_latent=None,
        noise=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            pcd = noise
        else:
            pcd = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    pcd,
                    t,
                    anchors=anchors,
                    code=code,
                    pointwise_latent=pointwise_latent
                )
                yield i, out
                pcd = out["sample"]
                
                
    def _vb_terms_bpd(
        self, x_start, x_t, t, code, model_output=None
    ):
        """
        Get a term for the variational lower-bound.
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            x_t, t, code, model_output=model_output
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -gaussian_log_likelihood(
            x_start, mean=out["mean"], logvar=out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    """
    def training_losses(self, x_start, t, code=None, pointwise_latent=None, noise=None):
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [B x 3 x N] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        if self.include_anchor_latent:
            assert pointwise_latent is not None 
            ctx = pointwise_latent
        elif self.include_global_latent:
            assert code is not None
            ctx = code 
        else:
            ctx = None

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                x_start=x_start,
                x_t=x_t,
                t=t,
                code=code,
                clip_denoised=False,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = self.model(x_t, self._scale_timesteps(t), ctx)

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["recons_loss"] = terms["mse"] + terms["vb"]
            else:
                terms["recons_loss"] = terms["mse"] if not self.scaled_loss else extract_into_tensor( self.betas * self.betas / (2 * (1. - self.betas) * (1. - self.alphas_cumprod)), t, terms["mse"].shape) * terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)
        
        return terms
    """
    def training_losses(self, x_start, t, anchors=None, code=None, pointwise_latent=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [B x 3 x N] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        if self.include_anchors:
            input = torch.cat([x_t, anchors], dim=1)
        else:
            input = x_t
        if self.include_anchor_latent:
            assert pointwise_latent is not None 
            ctx = pointwise_latent
        elif self.include_global_latent:
            assert code is not None
            ctx = code 
        else:
            ctx = None

        model_output = self.model(input, self._scale_timesteps(t), ctx)

        target = noise
        assert model_output.shape == target.shape == x_start.shape
        mse_loss = mean_flat((target - model_output) ** 2).mean()
        if self.scaled_loss:
            mse_loss = mse_loss * extract_into_tensor( self.betas * self.betas / (2 * (1. - self.betas) * (1. - self.alphas_cumprod)), t, mse_loss.shape)
        return dict(diffusion_loss=mse_loss)