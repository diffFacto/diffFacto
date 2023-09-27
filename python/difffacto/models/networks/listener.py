from distutils.command.build import build
import torch 
import torch.nn as nn 
from difffacto.utils.registry import MODELS, ENCODERS, METRICS, DECOMPOSERS, build_from_cfg
import pickle
import os
VOCAB_SIZE=2787
part_names = ["back", "seat", "leg", "arm"]

@MODELS.register_module()
class PartglotClassifier(nn.Module):
    def __init__(self, text_dim, embedding_dim, classifier_encoder, pcd_encoder, pcd_mixer, cross_attention, loss, num_part_latent, pcd_encoder_ckpt_path,pcd_mixer_ckpt_path,  out_dim):
        super().__init__()
        word2int = pickle.load(open("/mnt/disk3/wang/diffusion/anchorDIff/python/difffacto/models/networks/language_utils/word2int.pkl", "rb"))
        pn_tensor = torch.tensor([word2int[x] for x in part_names])[None]
        self.register_buffer("pn_tensor", pn_tensor)
        self.clsf_encoder = build_from_cfg(classifier_encoder, ENCODERS, vocab_size=VOCAB_SIZE)
        self.attn_encoder = nn.Sequential(self.clsf_encoder.word_embedding, nn.Linear(embedding_dim, text_dim))
        self.pcd_encoder = build_from_cfg(pcd_encoder, ENCODERS, num_anchors=num_part_latent)
        self.pcd_mixer = build_from_cfg(pcd_mixer, DECOMPOSERS, num_anchors=num_part_latent, point_dim=3)
        self.cross_attention = build_from_cfg(cross_attention, ENCODERS)
        self.loss = build_from_cfg(loss, METRICS)
        self.pcd_encoder.load_state_dict(torch.load(pcd_encoder_ckpt_path))
        # freeze pcd encoder
        for p in self.pcd_encoder.parameters():
            p.requires_grad=False
        self.pcd_mixer.load_state_dict(torch.load(pcd_mixer_ckpt_path))
        # freeze pcd mixer
        for p in self.pcd_mixer.parameters():
            p.requires_grad=False

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 128, bias=True), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True)
        )
        
    def forward(self, data, device='cuda'):
        # torch.save(self.clsf_encoder.state_dict(), "/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_anchor_feat_listener/checkpoints/classifier_encoder.pth")
        # torch.save(self.attn_encoder.state_dict(), "/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_anchor_feat_listener/checkpoints/attention_encoder.pth")
        # torch.save(self.mlp.state_dict(), "/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_anchor_feat_listener/checkpoints/listener_mlp.pth")
        # exit()
        pn_tensor = self.pn_tensor.to(device)
        target_pcds = data['target'].to(device)
        target_attn_maps = data['target_attn_map'].to(device)
        distractor_pcds = data['distractor'].to(device)
        distractor_attn_maps = data['distractor_attn_map'].to(device)
        part_indicators = data['part_indicator'].to(device).repeat_interleave(2, dim=0)
        texts = data['text'].to(device)
        B, npoint = target_pcds.shape[:-1]
        pcds = torch.stack([target_pcds, distractor_pcds], dim=1).reshape(B*2, npoint, 3)
        attn_maps = torch.stack([target_attn_maps, distractor_attn_maps], dim=1).reshape(B*2, npoint, -1)
        input = torch.cat([pcds, attn_maps], dim=-1)
        pcd_f = self.pcd_encoder(input)
        _, _, part_latent = self.pcd_mixer(pcd_f)

        attn_enc_f = self.attn_encoder(pn_tensor).repeat(B*2, 1, 1)

        clsf_enc_f, clsf_enc_attn = self.clsf_encoder(texts)
        clsf_enc_f = clsf_enc_f.repeat_interleave(2, dim=0).unsqueeze(1)

        cross_attn_f = self.cross_attention(attn_enc_f, part_latent, part_indicators)
        logits = self.mlp(torch.cat([clsf_enc_f, cross_attn_f], -1).squeeze(1)).reshape(B, 2)
        targets = torch.zeros(B, dtype=torch.int64).to(device)
        loss = self.loss(logits, targets)
        preds = torch.max(logits, 1)[1]
        num_correct = int(preds.shape[0] - preds.sum())
        if self.training:
            return dict(loss=loss) 
        
        return dict(
            logits=logits,
            num_correct=num_correct, 
            target=target_pcds, 
            distractor=distractor_pcds, 
            target_attn_map=target_attn_maps, 
            distractor_attn_map=distractor_attn_maps, 
            text=texts, 
            target_shift=data['target_shift'], 
            target_scale=data['target_scale'], 
            distractor_shift=data['distractor_shift'], 
            distractor_scale=data['distractor_scale'],
        )