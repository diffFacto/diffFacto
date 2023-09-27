import torch 
from torch.nn import Module
from difffacto.utils.registry import DIFFUSIONS, NETS, build_from_cfg
from difffacto.utils.constants import ModelMeanType, ModelVarType, LossType
from difffacto.utils.misc import *
from difffacto.utils.constants import *
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import numpy as np
from .diffusion_utils import extract_into_tensor, betas_for_alpha_bar, from_numpy_to_device, get_cov_from_params, normal_kl

@DIFFUSIONS.register_module()
class AnchoredDiffusion(Module):

    def __init__(
        self, 
        net, 
        num_timesteps, 
        beta_1, 
        beta_T, 
        k=1.,
        res=True,
        mode='linear', 
        use_beta=True, 
        rescale_timesteps=False, 
        loss_type='mse', 
        model_mean_type='epsilon',
        model_var_type='fixed_small', 
        scale_loss=False,
        clip_xstart=False,
        include_anchors=True,
        include_cov=False,
        learn_anchor=True,
        learn_variance=False,
        classifier_weight=1.,
        guidance=False,

        ddim_sampling=False,
        ddim_nsteps=10,
        ddim_discretize='uniform',
        ddim_eta=1.
    ):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.clip_xstart=clip_xstart
        self.res=res
        self.model = build_from_cfg(net, NETS)
        self.model_mean_type = model_mean_dict[model_mean_type]
        self.model_var_type = model_var_dict[model_var_type]
        self.loss_type = model_loss_dict[loss_type]
        self.scale_loss = scale_loss
        self.learn_anchor=learn_anchor
        self.learn_variance=learn_variance
        self.include_cov=include_cov 
        self.ddim_sampling=ddim_sampling
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.use_beta = use_beta
            
        self.rescale_timesteps = rescale_timesteps
        self.num_timesteps = int(num_timesteps)
        if mode == 'linear':
            betas = np.linspace(self.beta_1, self.beta_T, num=self.num_timesteps, dtype=np.float64)
        elif mode == 'cosine':
            betas = betas_for_alpha_bar(self.num_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
            
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.classifier_weight=classifier_weight
        self.guidance=guidance
        

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        if isinstance(k, list):
            self.k = np.array(k)
        else:
            self.k = np.array([k] * 3).astype(np.float32)
        self.k_squared = self.k
        self.log_k = np.log(self.k)
        self.include_anchors=include_anchors
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
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef3 = (
            1.0 + ((np.sqrt(self.alphas_cumprod) - 1.) * ( np.sqrt(self.alphas_cumprod_prev) + np.sqrt(alphas)))
            / (1.0 - self.alphas_cumprod)
        )
        
        if self.ddim_sampling:
            self.ddim_eta = ddim_eta
            self.xt_dir_coeff = np.sqrt(1. - self.alphas_cumprod - ddim_eta * ddim_eta * self.posterior_variance)
            if ddim_discretize == 'uniform':
                skip = self.num_timesteps // ddim_nsteps
                self.steps = list(range(0, self.num_timesteps, skip))
            elif ddim_discretize == 'quad':
                self.steps = (np.linspace(0., math.sqrt(self.num_timesteps * 0.8), ddim_nsteps) ** 2).astype(np.int32)
                self.steps = self.steps.tolist()
            else:
                NotImplementedError(ddim_discretize)
        else:
            self.steps = list(range(self.num_timesteps))

        
        
    # def q_mean_variance(self, x_start, t, anchors):
    #     """
    #     ## NOT USED WRONG
    #     Get the distribution q(x_t | x_0).
    #     :param x_0: the [N x 3 x n] tensor of noiseless inputs.
    #     :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    #     :param x_0: the [N x 3 x n] tensor of anchors the tensor of shape as x_0 to which x_0's diffuses
    #     :return: A tuple (mean, variance, log_variance), all of x_start's shape.
    #     """
    #     mean = (
    #         extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (x_start - anchors) + anchors
    #     )
    #     variance = extract_into_tensor((1.0 - self.alphas_cumprod), t, x_start.shape) * from_numpy_to_device(self.k_squared, x_start.device).reshape(1, 3, 1)
    #     log_variance = extract_into_tensor(
    #         self.log_one_minus_alphas_cumprod, t, x_start.shape
    #     ) + 2 * from_numpy_to_device(self.log_k, x_start.device).reshape(1, 3, 1)
    #     return mean, variance, log_variance
    
    def q_sample(self, x_start, t, anchors, noise=None, variance=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_0: the initial data batch.
        :param anchors: anchors for each point in x_0
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_0.
        """
        if self.learn_variance:
            assert variance is not None 
            assert variance.shape == anchors.shape
        if not self.learn_anchor:
            anchors = anchors * 0.0
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        B, C, N = x_start.shape
        if not self.learn_variance:
            variance = from_numpy_to_device(self.k_squared, x_start.device).reshape(1, 3, 1).expand(B, -1, N)
        L = torch.sqrt(variance)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (x_start - anchors) + anchors
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * L * noise)
        
    def q_posterior_mean(self, x_start, x_t, t, anchors):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape == anchors.shape
        if not self.learn_anchor:
            anchors = anchors * 0.0
        
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t 
            + extract_into_tensor(self.posterior_mean_coef3, t, anchors.shape) * anchors
        )
        assert (
            posterior_mean.shape[0]
            == x_t.shape[0]
        )
        return posterior_mean
    
    def q_posterior_variance(self, x_start, x_t, t, anchors, variance):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape == anchors.shape == variance.shape

        B, C, N = x_start.shape

        posterior_variance = extract_into_tensor(self.posterior_variance, t, variance.shape) * variance
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, variance.shape
        ) + torch.log(variance)
        assert (
            posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_t.shape[0]
        )
        return posterior_variance, posterior_log_variance_clipped
    
    def q_posterior_mean_variance(self, x_start, x_t, t, anchors, variance):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape == anchors.shape == variance.shape
        posterior_mean = self.q_posterior_mean(x_start, x_t, t, anchors)
        posterior_variance, posterior_log_variance_clipped = self.q_posterior_variance(x_start, x_t, t, anchors, variance=variance)
        # print("variance",  posterior_variance.permute([0, 3, 1,2]).reshape(-1, 3, 3))
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    
    def p_mean_variance(
        self, x, t, anchors, ctx=None, variance=None ,frozen_out=None, anchor_assignment=None, valid_id=None
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
        B, C, N = x.shape
        assert t.shape == (B,)
        if self.res:
            input = x - anchors 
        else:
            input = x
        if self.include_anchors:
            input = torch.cat([input, anchors], dim=1)
        if self.include_cov:
            input = torch.cat([input, variance], dim=1)
        
        if frozen_out is not None:
            model_output_cond = frozen_out
        else:
            model_output_cond = self.model(input, self._scale_timesteps(t), ctx, anchors=anchors.transpose(1,2), anchor_assignment=anchor_assignment, variances=variance.transpose(1,2), valid_id=valid_id)

        if self.guidance:
            uncond_ctx = [torch.zeros_like(r) for r in ctx]
            model_output_uncond = self.model(input, self._scale_timesteps(t), uncond_ctx, anchors=anchors.transpose(1,2), anchor_assignment=anchor_assignment, variances=variance.transpose(1,2), valid_id=valid_id)
            model_output = (1. - self.classifier_weight) * model_output_uncond + self.classifier_weight * model_output_cond
        else:
            model_output = model_output_cond

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C*2, N)
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
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

        if self.learn_variance:
            assert variance is not None 
            assert variance.shape == anchors.shape
        
        if not self.learn_variance:
            variance = from_numpy_to_device(self.k_squared, x.device).reshape(1, 3, 1).expand(B, -1, N)
        
        L = torch.sqrt(variance)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            model_sqrt_log_variance = 0.5 * model_log_variance.reshape(B, C, N) + torch.log(L)
            model_variance = model_variance.reshape(B, C, N) * variance
            model_log_variance = model_log_variance.reshape(B, C, N) + torch.log(variance)
        else:
            model_sqrt_log_variance = 0.5 * extract_into_tensor(model_log_variance, t, [B, 3, N]) + torch.log(L)
            model_variance = extract_into_tensor(model_variance, t, [B, 3, N]) * variance
            model_log_variance = extract_into_tensor(model_log_variance, t, [B, 3, N]) + torch.log(variance)
        

        def process_xstart(x):
            if self.clip_xstart:
                return x.clip(-10, 10)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, anchors=anchors, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [
            ModelMeanType.DRIFTED_EPSILON5, 
            ModelMeanType.SCALED_EPSILON, 
            ModelMeanType.START_X, 
            ModelMeanType.EPSILON, 
            ModelMeanType.EPSILON_AND_ANCHOR, 
            ModelMeanType.DRIFTED_EPSILON1, 
            ModelMeanType.DRIFTED_EPSILON2, 
            ModelMeanType.DRIFTED_EPSILON3, 
            ModelMeanType.DRIFTED_EPSILON4
            ]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            elif self.model_mean_type in [ModelMeanType.DRIFTED_EPSILON1, ModelMeanType.DRIFTED_EPSILON2, ModelMeanType.DRIFTED_EPSILON3, ModelMeanType.DRIFTED_EPSILON4]:
                if self.model_mean_type == ModelMeanType.DRIFTED_EPSILON1:
                    model_output = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, model_output.shape) * model_output
                elif self.model_mean_type == ModelMeanType.DRIFTED_EPSILON3:
                    model_output = extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, model_output.shape) * model_output
                elif self.model_mean_type == ModelMeanType.DRIFTED_EPSILON4:
                    model_output = model_output - anchors
                elif self.model_mean_type == ModelMeanType.DRIFTED_EPSILON5:
                    model_output = L * extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, model_output.shape) * model_output
                
                pred_xstart = process_xstart(
                    self._predict_xstart_from_drifted_eps(x_t=x, t=t, anchors=anchors, eps=model_output, sqrt_variance=L)
                )
            elif self.model_mean_type == ModelMeanType.EPSILON_AND_ANCHOR:
                # print(model_output.shape)
                pred_output, pred_anchor = torch.split(model_output, 3, dim=1)
                # print(pred_output.shape, pred_anchor.shape)
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, anchors=pred_anchor, eps=pred_output, sqrt_variance=L)
                )
            elif self.model_mean_type == ModelMeanType.SCALED_EPSILON:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, anchors=anchors, eps=model_output, sqrt_variance=None)
                )
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, anchors=anchors, eps=model_output, sqrt_variance=L)
                )
            if self.ddim_sampling:
                # print(xt_dir_coeff.mean())
                # print(model_variance.mean())
                # print(extract_into_tensor(self.alphas_cumprod_prev, timesteps=t, broadcast_shape=model_output.shape).mean())
                # print()
                xt_dir = L * extract_into_tensor(self.xt_dir_coeff, t, model_output.shape) * model_output
                ddim_variance = model_sqrt_log_variance 
            else:
                xt_dir=None
                ddim_variance=None
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t, anchors=anchors, variance=variance
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "sqrt_log_variance":model_sqrt_log_variance,
            "xt_dir": xt_dir,
            "ddim_variance":ddim_variance
        }
    
    def _predict_xstart_from_drifted_eps(self, x_t, t, anchors, eps, sqrt_variance):
        assert x_t.shape == eps.shape == anchors.shape
        
        return extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * (x_t - eps)
    def _predict_xstart_from_eps(self, x_t, t, anchors, eps, sqrt_variance):
        assert x_t.shape == eps.shape == anchors.shape
        if sqrt_variance is None:
            sqrt_variance = torch.ones_like(x_t)
        if not self.learn_anchor:
            anchors = anchors * 0.0
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * (x_t - anchors) + anchors
                -  extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, sqrt_variance.shape) * sqrt_variance * eps
                )

    def _predict_xstart_from_xprev(self, x_t, t, anchors, xprev):
        assert x_t.shape == xprev.shape == anchors.shape
        if not self.learn_anchor:
            anchors = anchors * 0.0
        return (  # (xprev - coef2*x_t - coef3 * anchors) / coef1
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            ) * x_t
            - extract_into_tensor(
                self.posterior_mean_coef3 / self.posterior_mean_coef1, t, x_t.shape
            ) * anchors
        )
    
    def _predict_eps_from_xstart(self, x_t, t, anchors, pred_xstart, variance=None):
    # Not WOrking !!!!! Don't Use!!!!
        B, C, N = x_t.shape

        if self.learn_variance:
            assert variance is not None 
            assert variance.shape == anchors.shape
        if not self.learn_anchor:
            anchors = anchors * 0.0
            
        if not self.learn_variance:
            variance = from_numpy_to_device(self.k_squared, x_t.device).reshape(1, 3, 1).expand(B, -1, N)
        L = torch.sqrt(variance)
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * (x_t - anchors)
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, L.shape)
        
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        if self.use_beta:
            return extract_into_tensor(self.betas, t, t.shape)
        return t

    def p_sample(
        self, x, t, anchors, ctx=None, variance=None, anchor_assignment=None, valid_id=None
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
            anchors,
            ctx=ctx,
            variance=variance,
            anchor_assignment=anchor_assignment,
            valid_id=valid_id
        )
        if not self.learn_anchor:
            anchors = anchors * 0.0
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if self.ddim_sampling:
            sample = (out["pred_xstart"] - anchors) * torch.sqrt(extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)) + anchors + out['xt_dir'] + self.ddim_eta * nonzero_mask * torch.sqrt(out["variance"]) * noise
        else:
            sample = out["mean"] + nonzero_mask * torch.sqrt(out["variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def p_sample_loop(
        self,
        shape,
        anchors,
        ctx=None,
        noise=None,
        variance=None,
        anchor_assignment=None,
        valid_id=None,
        device=None,
        progress=False
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
            ctx=ctx,
            variance=variance,
            anchor_assignment=anchor_assignment,
            valid_id=valid_id,
            device=device,
            progress=progress,
        ):
            final = sample

        return final["sample"]

    def p_sample_loop_progressive(
        self,
        shape,
        anchors,
        ctx=None,
        variance=None,
        anchor_assignment=None,
        valid_id=None,
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
            B, _, N = shape

            if self.learn_variance:
                assert variance is not None 
                assert variance.shape == anchors.shape

            if not self.learn_variance:
                cov = from_numpy_to_device(self.k_squared, device).reshape(1, 3, 1).expand(B, -1, N)
            else:
                cov = variance
            L = torch.sqrt(cov)
            pcd = L  * torch.randn(*shape, device=device) + anchors * int(self.learn_anchor)

        indices = self.steps[::-1]
        print("sampling trajectory: %s" % [s for s in indices])

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        yield self.num_timesteps, dict(sample=pcd)
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    pcd,
                    t,
                    anchors,
                    ctx=ctx,
                    variance=variance,
                    anchor_assignment=anchor_assignment,
                    valid_id=valid_id
                )
                yield i, out
                pcd = out["sample"]
                
    def q_sample_loop(
        self,
        gt_pcd,
        anchors,
        variance=None,
        noise=None,
        device=None,
        progress=False,
    ):
        """
        forward sample from the model
        :param shape: the shape of the samples, (B, C, N).
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
        for t, sample in self.q_sample_loop_progressive(
            gt_pcd,
            noise=noise,
            anchors=anchors,
            variance=variance,
            device=device,
            progress=progress,
        ):
            final = sample

        return final

    def q_sample_loop_progressive(self, gt_pcd, anchors, noise=None, variance=None, device=None, progress=False):
        if device is None:
            device = next(self.model.parameters()).device
        indices = list(range(self.num_timesteps))[1:]
        if noise is None:
            noise = torch.randn(*gt_pcd.shape).to(device)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = torch.tensor([i] * gt_pcd.shape[0], device=device)
            with torch.no_grad():
                yield i, self.q_sample(gt_pcd, t, anchors, noise=noise, variance=variance)

    def _vb_terms_bpd(
        self, x_start, x_t, t, anchors, ctx=None, variance=None, frozen_out=None, anchor_assignment=None, flags=None
    ):
        """
        Get a term for the variational lower-bound.
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        B, C, N = x_start.shape
        
        true_mean, true_posterior_variance, _ = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t, anchors=anchors, variance=variance
        )
        true_posterior_diag_var = torch.diagonal(true_posterior_variance.permute([0, 3, 1, 2]).reshape(B*N, C, C), offset=0, dim1=-2, dim2=-1)
        log_true_posterior_diag_var = torch.log(true_posterior_diag_var)
        out = self.p_mean_variance(
            x_t, t, anchors, ctx=ctx, variance=variance, frozen_out=frozen_out, anchor_assignment=anchor_assignment
        )
        # print(true_mean.shape, out["mean"].shape, true_log_variance_clipped.shape, out["log_variance"].shape)
        # print(true_posterior_variance)
        kl = normal_kl(
            true_mean.transpose(1,2).reshape(B*N, C), log_true_posterior_diag_var, out["mean"].transpose(1,2).reshape(B*N, C), out["log_variance"].transpose(1,2).reshape(B*N, C), dim=C
        ).sum(-1)
        # print(kl.reshape(B, N).mean(-1))
        # print()
        if flags is not None:
            kl = (kl.reshape(B, 1, N) * flags).sum(dim=(1,2)) / (flags.sum(dim=(1,2)) * np.log(2.0))
        else:
            kl = kl.reshape(B, N).mean(-1) / np.log(2.0)
        # print(torch.all(torch.det(out["variance"].permute([0, 3, 1, 2]).reshape(B*N, C, C)) > 0))
        decoder_nll = -gaussian_log_likelihood(z=x_start.transpose(1,2).reshape(B*N, C), mean=out["mean"].transpose(1,2).reshape(B*N, C), logvar=out["log_variance"].transpose(1,2).reshape(B*N, C), dim=C).sum(-1)
        # print(decoder_nll.reshape(B, N).mean(-1))
        # print(t)
        if flags is not None:
            decoder_nll = (decoder_nll.reshape(B, 1, N) * flags).sum(dim=(1,2)) / (flags.sum(dim=(1,2)) * np.log(2.0))
        else:
            decoder_nll = decoder_nll.reshape(B, N).mean(-1) / np.log(2.0)
        
            
        # if kl > 1000:
            # print(decoder_nll, kl, out["mean"], out["variance"].permute([0, 3, 1, 2]).reshape(B*N, C, C), t)
            # exit()
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    """
    def training_losses(self, x_start, t, anchors, x_start_feats, code=None, partial_pc=None, noise=None):
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [B x 3 x N] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, anchors, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                x_start=x_start,
                x_t=x_t,
                t=t,
                anchors=anchors,
                x_start_feats=x_start_feats,
                code=code,
                clip_denoised=False,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = self.model(x_t, self._scale_timesteps(t), x_start_feats, anchors, code, partial_pc)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    anchors=anchors, 
                    x_start_feats=x_start_feats,
                    code=code,
                    model_output=frozen_out
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t, anchors=anchors
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["recons_loss"] = terms["mse"] + terms["vb"]
            else:
                terms["recons_loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)
        return terms["recons_loss"].mean()
    """
    def training_losses(self, x_start, t, anchors=None, variance=None, ctx=None, reduce=True, anchor_assignment=None, valid_id=None, flags=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [B x 3 x N] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        loss_dict = dict()
        if anchors is None:
            anchors = torch.zeros_like(x_start)
        if self.learn_variance:
            assert variance is not None 
            assert variance.shape == anchors.shape
            # print(variance.mean())

        if noise is None:
            noise = torch.randn_like(x_start)
        B, C, N = x_start.shape
        ori_anchors = anchors
        if not self.learn_anchor:
            anchors = anchors * 0.0
        x_t = self.q_sample(x_start, t, anchors, noise=noise, variance=variance)
        model_in=x_t
        if self.res:
            model_in = model_in - ori_anchors
        if self.include_anchors:
            model_in = torch.cat([model_in, ori_anchors], dim=1)
        if self.include_cov:
            model_in = torch.cat([model_in, variance], dim=1)

        
        model_output = self.model(model_in, self._scale_timesteps(t), ctx, anchor_assignment=anchor_assignment, anchors=ori_anchors.transpose(1,2), variances=variance.transpose(1,2) if variance is not None else None, valid_id=valid_id)
        if self.model_var_type in [
            ModelVarType.LEARNED,
            ModelVarType.LEARNED_RANGE,
        ]:
            # print(model_output.shape)
            assert model_output.shape == (B, C*2, N)
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            loss_dict['model_var_value'] = model_var_values.mean()
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            vb_loss= self._vb_terms_bpd(
                x_start=x_start,
                x_t=x_t,
                t=t,
                anchors=ori_anchors,
                ctx=ctx,
                variance=variance,
                frozen_out=frozen_out,
                anchor_assignment=anchor_assignment,
                flags=flags
            )["output"].mean()
            # print(vb_loss)
            if vb_loss > 1000:
                print("vb_loss is", vb_loss.mean())
                print("model var is ", model_var_values.mean())
                exit()
            vb_loss *= self.num_timesteps / 1000.0
            loss_dict['vb_loss'] = vb_loss
            
        if not self.learn_variance:
            variance = from_numpy_to_device(self.k_squared, x_start.device).reshape(1, 3, 1).expand(B, -1, N)
        L = torch.sqrt(variance)
        target = {
                ModelMeanType.START_X: x_start,
                ModelMeanType.DRIFTED_EPSILON1: extract_into_tensor(1. - self.sqrt_alphas_cumprod, t, noise.shape) * anchors + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise,
                ModelMeanType.DRIFTED_EPSILON2: extract_into_tensor(self.sqrt_recip_alphas_cumprod - 1, t, noise.shape) * anchors + extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, noise.shape) * noise,
                ModelMeanType.DRIFTED_EPSILON3: extract_into_tensor((1. - self.sqrt_alphas_cumprod) / self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * anchors + L * noise,
                ModelMeanType.DRIFTED_EPSILON4: extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, noise.shape) * anchors + extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, noise.shape) * noise,
                ModelMeanType.DRIFTED_EPSILON5: extract_into_tensor((1. - self.sqrt_alphas_cumprod) / self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * anchors / L + noise,
                ModelMeanType.SCALED_EPSILON: L * noise,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_AND_ANCHOR: torch.cat([noise, anchors], dim=1)
            }[self.model_mean_type]
        assert model_output.shape == target.shape
        diffusion_loss = (target - model_output) ** 2
        if flags is not None:
            diffusion_loss = diffusion_loss * flags
        if reduce:
            if flags is not None:
                diffusion_loss = diffusion_loss.mean(1).sum() / flags.sum()
            else:
                diffusion_loss = diffusion_loss.mean()
            
        if self.scale_loss:
            diffusion_loss = diffusion_loss * extract_into_tensor( self.betas * self.betas / (2 * (1. - self.betas) * (1. - self.alphas_cumprod)), t, diffusion_loss.shape)
        loss_dict['mse_loss'] = diffusion_loss
        
        return loss_dict