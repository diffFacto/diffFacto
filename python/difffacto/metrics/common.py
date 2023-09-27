import torch
import torch.nn.functional as F
from difffacto.utils.registry import METRICS
import math 
import random

def triplet_loss(anchor, pos, neg, thresh=0.1):
    pos_dist = (anchor - pos) ** 2
    neg_dist = (anchor - neg) ** 2
    l = pos_dist - neg_dist + thresh
    l = l.mean(1)
    return torch.clamp(l, min=0), pos_dist.mean(), neg_dist.mean()


@METRICS.register_module()
def dis_loss(d_real, d_fake, loss_type="wgan", weight=1., **kwargs):
    if loss_type.lower() == "wgan":
        loss_fake = d_fake.mean()
        loss_real = d_real.mean()
        wg_loss_orig = loss_fake - loss_real
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_dis_l": wg_loss.clone().detach(),
            "wgan_dis_l_orig": wg_loss_orig.clone().detach(),
            "wgan_dis_l_real": loss_real.clone().detach(),
            "wgan_dis_l_fake": loss_fake.clone().detach()
        }
    elif loss_type.lower() == "hinge":
        d_loss_real = -torch.min(d_real - 1, d_real * 0).mean()
        d_loss_fake = -torch.min(-d_fake - 1, d_fake * 0).mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        d_loss = d_loss_real + d_loss_fake
        loss = d_loss * weight
        return loss, {
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            "l_real": d_loss_real.clone().detach(),
            "l_fake": d_loss_fake.clone().detach(),
        }
    else:
        raise NotImplementedError("Not implement: %s" % loss_type)


@METRICS.register_module()
def gradient_penalty(x_real, x_fake, d_real, d_fake,
                     weight=1., gp_type='zero_center', interpolated=None, d_interpolated=None, seps=1e-8):
    if gp_type == "zero_center":
        bs = d_real.size(0)
        grad = torch.autograd.grad(
            outputs=d_real, inputs=x_real,
            grad_outputs=torch.ones_like(d_real).to(d_real),
            create_graph=True, retain_graph=True)[0]
        # [grad] should be either (B, D) or (B, #points, D)
        grad = grad.reshape(bs, -1)
        grad_norm = gp_orig = torch.sqrt(torch.sum(grad ** 2, dim=1)).mean()
        gp = gp_orig ** 2. * weight
        return gp, {
            'gp': gp.clone().detach().cpu(),
            'gp_orig': gp_orig.clone().detach().cpu(),
            'grad_norm': grad_norm.clone().detach().cpu()
        }
    elif gp_type == "interpolated":
        bs = d_real.size(0)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(d_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(bs, -1)

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + seps)

        # Return gradient penalty
        gp =  weight * ((gradients_norm - 1) ** 2).mean()
        return gp, {
            'gp': gp.clone().detach().cpu(),
            'grad_norm': gradients_norm.mean().clone().detach().cpu()
        }
    else:
        raise NotImplemented("Invalid gp type:%s" % gp_type)

@METRICS.register_module()
def gen_loss(d_real, d_fake, loss_type="wgan", weight=1., **kwargs):
    if loss_type.lower() == "wgan":
        wg_loss_orig = - d_fake.mean()
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_gen_l_orig": wg_loss_orig.clone().detach(),
        }
    elif loss_type.lower() == "hinge":
        g_loss = -d_fake.mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        loss = weight * g_loss
        return loss, {
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
        }

    else:
        raise NotImplementedError("Not implement: %s" % loss_type)
    
@METRICS.register_module()
class SmoothCrossEntropy:
    # https://arxiv.org/pdf/1512.00567.pdf
    def __init__(self, alpha=0.1):
        self.alpha=alpha
    def __call__(self, pred, target, ):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)
        one_hot = (
            one_hot * ((1.0 - self.alpha) + self.alpha / n_class) + (1.0 - one_hot) * self.alpha / n_class
        )  # smoothed

        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return torch.mean(loss)
    
@METRICS.register_module()
class NegativeGaussianLogLikelihood:
    def __init__(self, dim=3, reduction='mean'):
        self.dim=dim 
        self.log_z = 0.5 * dim * math.log(2 * math.pi)
        self.reduction=reduction
    def __call__(self, z, mu, var=None):
        assert z.shape[-1] == mu.shape[-1] == self.dim
        if isinstance(var, float):
            var_term = 0
            var = 1.
        else:
            var_term = 0.5 * torch.sum(torch.log(var), dim=-1)
        
        if self.reduction == 'mean':
            return self.log_z + var_term + ((z - mu) ** 2 / var).mean() / 2
        elif self.reduction is None:
            return self.log_z + var_term + ((z - mu) ** 2 / var) / 2


@METRICS.register_module()
class CrossEntropy:
    def __init__(self, n_class=2, reduce=True):
        self.reduce=reduce
        self.n_class=n_class
    def __call__(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        target_prob = torch.sigmoid(target)
        pred_dist = torch.stack([pred_prob, 1 - pred_prob], dim=1)
        target_dist = torch.stack([target_prob, 1 - target_prob], dim=1)

        log_prb = torch.log(pred_dist)
        loss = -(target_dist * log_prb).sum(dim=1)
        return torch.mean(loss) if self.reduce else loss

@METRICS.register_module()
class L2Loss:
    def __init__(self, reduction='mean'):
        self.reduction=reduction
    def __call__(self, source, target, var=None):
        if self.reduction == 'mean':
            diff = torch.pow(source - target, 2).mean()
        elif self.reduction is None:
            diff = torch.pow(source - target, 2).mean(-1)
        return diff

@METRICS.register_module()
class L1Loss:
    def __init__(self, reduction='mean'):
        self.reduction=reduction
    def __call__(self, source, target):
        return torch.abs(source - target).mean()


@METRICS.register_module()
class SpectralContrastiveLoss:
    def __init__(self):
        pass
    def __call__(self, input):
        anchor, pos, neg = input[:, 0], input[:, 1], input[:, 2] # [B, dim]
        anchor, pos, neg = map(lambda t: F.normalize(t, 2, 1), [anchor, pos, neg])
        B, dim=anchor.shape
        anchor_pos_dot = torch.bmm(anchor.unsqueeze(1), pos.unsqueeze(2)).reshape(B)
        anchor_neg_dot = torch.bmm(anchor.unsqueeze(1), neg.unsqueeze(2)).reshape(B)
        l = -2. * anchor_pos_dot + anchor_neg_dot ** 2
        return l.mean()