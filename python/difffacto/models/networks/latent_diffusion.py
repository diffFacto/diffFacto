from distutils.command.build import build
import torch
import torch.nn as nn 
from torch.nn import Module, functional
from difffacto.utils.misc import *
from difffacto.utils.registry import build_from_cfg, MODELS, ENCODERS, DIFFUSIONS, SAMPLERS, METRICS, DECOMPOSERS, SEGMENTORS
from pointnet2_ops.pointnet2_utils import gather_operation
from .language_utils.language_util import tokenizing
VOCAB_SIZE=2787

@MODELS.register_module()
class LatentDiffEdit(Module):

    def __init__(
        self, 
        encoder, 
        diffusion, 
        sampler, 
        num_anchors,
        num_timesteps, 
        guidance=False,
        language_encoder=None,
        pcd_encoder_ckpt=None,
        pcd_mixer=None,
        pcd_mixer_ckpt=None,
        part_latent_dropout_prob=0.2,
        language_latent_dropout_prob=0.2,
        zero_part_latent=False,

        save_dir=None,
        save_weights=False
        ):
        super().__init__()
        self.encoder = build_from_cfg(encoder, ENCODERS, num_anchors=num_anchors)
        self.diffusion = build_from_cfg(diffusion, DIFFUSIONS, num_timesteps=num_timesteps, num_part=num_anchors)
        self.encoder.load_state_dict(torch.load(pcd_encoder_ckpt))
        for m in self.encoder.parameters():
            m.requires_grad = False
        self.sampler = build_from_cfg(sampler, SAMPLERS, num_timesteps=num_timesteps)
        self.language_encoder = build_from_cfg(language_encoder, ENCODERS, vocab_size=VOCAB_SIZE)
        self.pcd_mixer = build_from_cfg(pcd_mixer,DECOMPOSERS, num_anchors=num_anchors, point_dim=3)
        if self.pcd_mixer is not None:
            self.pcd_mixer.load_state_dict(torch.load(pcd_mixer_ckpt))
        for m in self.pcd_mixer.parameters():
            m.requires_grad = False
        self.zero_part_latent=zero_part_latent
        self.guidance=guidance
        self.part_latent_dropout_prob=part_latent_dropout_prob
        self.language_latent_dropout_prob=language_latent_dropout_prob
        self.save_weights=save_weights
        self.save_dir=save_dir
        self.num_timesteps = int(num_timesteps)
        self.num_anchors = num_anchors
        
    def decode(self, prior, code=None, part_indicator=None, anchors=None, noise=None, device='cuda'):
        final = dict() 
        for t, sample in self.diffusion.p_sample_loop_progressive(
            prior.shape,
            prior=prior,
            code=code,
            part_indicator=part_indicator,
            anchors=anchors,
            noise=noise,
            device=device,
            progress=False,
        ):
            if t == 0:
                final['pred'] = sample['sample'].transpose(2, 1)
        return final['pred']
        
    def forward(self, pcds, device='cuda', iter=-1):
        """
            input: [B, npoints, 3]
            attn_map: [B, npoints, n_class]
            seg_mask: [B, npoints]
        """
        if not self.training and self.save_weights:
            self.eval()
            os.system(f"mkdir -p {self.save_dir}")
            torch.save(self.language_encoder.state_dict(), os.path.join(self.save_dir, "language_encoder_ckpt.pth"))
            torch.save(self.diffusion.state_dict(), os.path.join(self.save_dir, "ldm_ckpt.pth"))
            return dict(pred=pcds['target'],distractor=pcds['target'], target=pcds['target'], shift=pcds['target_shift'], scale=pcds['target_scale'])
        target = pcds['target'].to(device)
        distractors = pcds['distractor'].to(device)
        target_attn_map = pcds['target_attn_map'].to(device)
        distractor_attn_map = pcds['distractor_attn_map'].to(device)
        part_indicator = pcds['part_indicator'].to(device)
        text = pcds['text'].to(device)
        
        loss_dict = dict()
               
        input_pcds = torch.stack([target, distractors], dim=1) # [B, 2, npoint, 3]
        input_attn_map = torch.stack([target_attn_map, distractor_attn_map], dim=1)
        x = torch.cat([input_pcds, input_attn_map], dim=-1) # -> [B, 2, npoints, n_class + 3]

        B, K, npoint = input_pcds.shape[:-1]
        part_latent = self.encoder(x.reshape(B*2, npoint, -1))
        if self.pcd_mixer is not None:
            _, anchors, _ = self.pcd_mixer(part_latent)
        if iter == 0:
            self.diffusion.on_train_batch_start(part_latent, iter)
        target_part_latent, distractor_part_latent = torch.split(part_latent.reshape(B, 2, self.num_anchors, -1), 1, dim=1)
        target_anchors, distractor_anchors = torch.split(anchors.reshape(B, 2, self.num_anchors, 3), 1, dim=1)
        language_part_latent, language_enc_attn = self.language_encoder(text)
        target_part_latent, distractor_part_latent, target_anchors, distractor_anchors = map(lambda t: t.squeeze(1).transpose(1,2), [target_part_latent, distractor_part_latent, target_anchors, distractor_anchors])
        if len(language_part_latent.shape) == 3:
            language_part_latent = language_part_latent.transpose(1,2)
        if self.training:
            if self.guidance:
                idxs = torch.rand(distractor_part_latent.shape[0], distractor_part_latent.shape[2]).to(distractor_part_latent.device) >= self.part_latent_dropout_prob
                distractor_part_latent = distractor_part_latent * idxs.unsqueeze(1)

                idxxs = torch.rand(language_part_latent.shape[0], language_part_latent.shape[2]).to(language_part_latent.device) >= self.language_latent_dropout_prob
                language_part_latent = language_part_latent * idxxs.unsqueeze(1)
            if self.zero_part_latent:
                distractor_part_latent = torch.zeros_like(distractor_part_latent)
            t, _ = self.sampler.sample(B, device)
            diffusion_loss = self.diffusion.training_losses(target_part_latent, t, prior=distractor_part_latent, code=language_part_latent, part_indicator=part_indicator, anchors=distractor_anchors)
            loss_dict.update(diffusion_loss)
            return loss_dict
        else:
            pred = self.decode(prior=language_part_latent, code=language_part_latent, part_indicator=part_indicator, anchors=distractor_anchors, device=device) 
            return pred
