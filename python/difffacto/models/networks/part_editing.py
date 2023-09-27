from distutils.command.build import build
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from difffacto.utils.registry import MODELS, ENCODERS, METRICS, DECOMPOSERS, build_from_cfg
import pickle
import os
from pointnet2_ops.pointnet2_utils import gather_operation
VOCAB_SIZE=2787
part_names = ["back", "seat", "leg", "arm"]

@MODELS.register_module()
class PartLatentEditor(nn.Module):
    def __init__(
        self, 
        text_dim, 
        embedding_dim, 
        part_latent_dim, 
        language_encoder, 
        latent_language_fuser,
        classifier_encoder, 
        pcd_encoder, 
        pcd_mixer, 
        cross_attention, 
        loss, 
        num_part_latent, 
        pcd_encoder_ckpt_path, 
        pcd_mixer_ckpt_path, 
        attn_encoder_ckpt_path, 
        classifier_encoder_ckpt_path, 
        listener_mlp_ckpt_path, 
        out_dim,
        direction_loss,
        magnitude_loss,
        logit_loss,
        logit_weight=1.0,
        direction_weight=1.0,
        magnitude_weight=1.0,
        supervise_before_mixer=False,
        icmle=False,
        conditional_dim=24,
        num_coditional_sample=10,
        save_weights=False,
        save_dir=None,
    ):
        super().__init__()
        self.part_latent_dim=part_latent_dim
        self.direction_loss=direction_loss
        self.magnitude_loss=magnitude_loss
        self.logit_loss=logit_loss
        self.num_coditional_sample=num_coditional_sample
        self.logit_weight=logit_weight
        self.direction_weight=direction_weight
        self.save_weights=save_weights
        self.icmle=icmle
        self.save_dir=save_dir
        self.conditional_dim=conditional_dim
        self.magnitude_weight=magnitude_weight
        self.supervise_before_mixer=supervise_before_mixer
        word2int = pickle.load(open("/mnt/disk3/wang/diffusion/anchorDIff/python/difffacto/models/networks/language_utils/word2int.pkl", "rb"))
        pn_tensor = torch.tensor([word2int[x] for x in part_names])[None]
        self.register_buffer("pn_tensor", pn_tensor)
        self.language_encoder = build_from_cfg(language_encoder, ENCODERS, vocab_size=VOCAB_SIZE)
        self.latent_language_fuser = build_from_cfg(latent_language_fuser, ENCODERS, num_part=num_part_latent, conditional=icmle, conditional_dim=conditional_dim)
        self.clsf_encoder = build_from_cfg(classifier_encoder, ENCODERS, vocab_size=VOCAB_SIZE)
        self.attn_encoder = nn.Sequential(self.clsf_encoder.word_embedding, nn.Linear(embedding_dim, text_dim))
        self.pcd_encoder = build_from_cfg(pcd_encoder, ENCODERS, num_anchors=num_part_latent)
        self.pcd_mixer = build_from_cfg(pcd_mixer, DECOMPOSERS, num_anchors=num_part_latent, point_dim=3)
        self.cross_attention = build_from_cfg(cross_attention, ENCODERS)
        self.loss = build_from_cfg(loss, METRICS)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 128, bias=True), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True)
        )

        # freeze mlp
        self.mlp.load_state_dict(torch.load(listener_mlp_ckpt_path))
        for p in self.mlp.parameters():
            p.requires_grad=False
        # freeze attn encoder
        self.attn_encoder.load_state_dict(torch.load(attn_encoder_ckpt_path))
        for p in self.attn_encoder.parameters():
            p.requires_grad=False
        # freeze classifier encoder
        self.clsf_encoder.load_state_dict(torch.load(classifier_encoder_ckpt_path))
        for p in self.clsf_encoder.parameters():
            p.requires_grad=False
        self.pcd_encoder.load_state_dict(torch.load(pcd_encoder_ckpt_path))
        # freeze pcd encoder
        for p in self.pcd_encoder.parameters():
            p.requires_grad=False
        self.pcd_mixer.load_state_dict(torch.load(pcd_mixer_ckpt_path))
        # freeze pcd mixer
        for p in self.pcd_mixer.parameters():
            p.requires_grad=False

        
        
    def forward(self, data, device='cuda'):
        if self.save_weights:
            torch.save(self.language_encoder.state_dict(), os.path.join(self.save_dir, "language_encoder.pth"))
            torch.save(self.latent_language_fuser.state_dict(), os.path.join(self.save_dir, "update_mlp.pth"))
            exit()
        loss_dict = dict()
        pn_tensor = self.pn_tensor.to(device)
        target_pcds = data['target'].to(device)
        target_attn_maps = data['target_attn_map'].to(device)
        distractor_pcds = data['distractor'].to(device)
        distractor_attn_maps = data['distractor_attn_map'].to(device)
        part_indicators = data['part_indicator'].to(device)
        texts = data['text'].to(device)
        B, npoint = target_pcds.shape[:-1]
        pcds = torch.stack([target_pcds, distractor_pcds], dim=1).reshape(B*2, npoint, 3)
        attn_maps = torch.stack([target_attn_maps, distractor_attn_maps], dim=1).reshape(B*2, npoint, -1)
        input = torch.cat([pcds, attn_maps], dim=-1)
        part_latent = self.pcd_encoder(input)
        

        n_part, n_dim = part_latent.shape[1:]
        tgt_latent, latent_for_edit = part_latent.reshape(B, 2, n_part, n_dim).split(1, dim=1)
        tgt_latent, latent_for_edit = tgt_latent.reshape(B, n_part, n_dim), latent_for_edit.reshape(B, n_part, n_dim)
        clsf_enc_f, clsf_enc_attn = self.clsf_encoder(texts)
        language_enc_f, language_enc_attn = self.language_encoder(texts)
        if self.icmle:
            K=self.num_coditional_sample
            conditional = torch.randn(B, K, self.conditional_dim).cuda()
        else:
            K=1
            conditional = None
        clsf_enc_f = clsf_enc_f.repeat_interleave(K, dim=0)
        tgt_latent = tgt_latent.unsqueeze(1).repeat_interleave(K, dim=1)
        edited_latent = self.latent_language_fuser(latent_for_edit, part_indicators, language_enc_f, conditional=conditional)
        
        part_id = part_indicators.repeat_interleave(2*K, dim=0).max(1)[1].to(torch.int)
        part_latent_pre_attn = torch.stack([tgt_latent, edited_latent], dim=1).reshape(B*2*K, n_part, n_dim)
        
        _, _, part_latent_attn_out = self.pcd_mixer(part_latent_pre_attn)
        if self.supervise_before_mixer:
            referenced_part_latent = gather_operation(part_latent_pre_attn.reshape(B*2*K, n_part, n_dim).transpose(1,2).contiguous(), part_id.reshape(B*2*K, 1)).reshape(B, 2, K, 1, -1)
        else:
            referenced_part_latent = gather_operation(part_latent_attn_out.reshape(B*2*K, n_part, n_dim).transpose(1,2).contiguous(), part_id.reshape(B*2*K, 1)).reshape(B, 2, K, 1, -1)
            
        tgt_latent, edited_latent = referenced_part_latent.reshape(B, 2, K, 1, n_dim).split(1, dim=1)
            
        tgt_latent, edited_latent = tgt_latent.reshape(B*K, n_dim), edited_latent.reshape(B*K, n_dim)
        loss = torch.zeros([B, K]).cuda()
        if self.direction_loss:
            cos_sim = F.cosine_similarity(tgt_latent, edited_latent, dim=1)
            dir_loss = 1 - cos_sim
            loss += dir_loss.reshape(B, K)
            loss_dict["direction_loss"] = self.direction_weight * dir_loss
        if self.magnitude_loss:
            tgt_norm, src_norm = torch.norm(tgt_latent, p=2, dim=1), torch.norm(edited_latent, p=2, dim=1)
            mag_loss = (tgt_norm - src_norm) ** 2
            loss += mag_loss.reshape(B, K)
            loss_dict["magnitude_loss"] = self.magnitude_weight * mag_loss
        if self.logit_loss:
            modified_logits = self.mlp(torch.cat([clsf_enc_f, edited_latent], -1))
            logits = self.mlp(torch.cat([clsf_enc_f, tgt_latent], -1))
            logit_loss = self.loss(modified_logits, logits)
            loss += logit_loss.reshape(B, K)
            loss_dict['logit_loss'] = self.logit_weight*logit_loss
            
        min_idx = loss.min(dim=1)[1]
        loss_dict = {k: torch.gather(v.reshape(B, K), 1, min_idx.unsqueeze(1)).mean() for k, v in loss_dict.items()}
        
        if self.training:
            return loss_dict

        return dict(
            num_correct=0,
            target=target_pcds, 
            distractor=distractor_pcds, 
            target_attn_map=target_attn_maps, 
            distractor_attn_map=distractor_attn_maps, 
            text=texts, 
            target_shift=data['target_shift'], 
            target_scale=data['target_scale'], 
            distractor_shift=data['distractor_shift'], 
            distractor_scale=data['distractor_scale'],
            modified_logits=modified_logits if self.logit_loss else torch.zeros(B, 2),
            logits=logits if self.logit_loss else torch.zeros(B, 2)
        )