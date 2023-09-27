import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from difffacto.utils.registry import ENCODERS
from pointnet2_ops.pointnet2_utils import gather_operation
import math

@ENCODERS.register_module()
class LSTM(nn.Module):
    def __init__(self, text_dim=64, embedding_dim=100, vocab_size=2787, padding_idx=0):
        super().__init__()
        self.text_dim=text_dim
        self.padding_idx = padding_idx
        self.word_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(embedding_dim, text_dim, batch_first=True)
        self.w_attn = nn.Parameter(torch.Tensor(1, text_dim))
        nn.init.xavier_uniform_(self.w_attn)

    def forward(self, padded_tokens, dropout=0.5):
        
        w_emb = self.word_embedding(padded_tokens)
        w_emb = F.dropout(w_emb, dropout, self.training)
        len_seq = (padded_tokens != self.padding_idx).sum(dim=1).cpu()
        x_packed = pack_padded_sequence(
            w_emb, len_seq, enforce_sorted=False, batch_first=True
        )

        B = padded_tokens.shape[0]

        rnn_out, _ = self.rnn(x_packed)
        rnn_out, dummy = pad_packed_sequence(rnn_out, batch_first=True)
        h = rnn_out[torch.arange(B), len_seq - 1]
        final_feat, attn = self.word_attention(rnn_out, h, len_seq)
        return final_feat, attn
        
        
        # w_emb = self.word_embedding(padded_tokens)
        # w_emb = F.dropout(w_emb, dropout, self.training)
        # len_seq = (padded_tokens != self.padding_idx).sum(dim=1).cpu()
        # x_packed = pack_padded_sequence(
        #     w_emb, len_seq, enforce_sorted=False, batch_first=True
        # )

        # B = padded_tokens.shape[0]

        # rnn_out, _ = self.rnn(x_packed)
        # rnn_out, dummy = pad_packed_sequence(rnn_out, batch_first=True)
        # h = rnn_out[torch.arange(B), len_seq - 1]
        # final_feat, attn = self.word_attention(rnn_out, h, len_seq)
        # if self.regress_part:
        #     regress_first, regress_rest = self.part_regressor[0], self.part_regressor[1:]
        #     part_f_in = regress_first(final_feat).reshape(B, -1, self.num_part)
        #     part_f_out = regress_rest(part_f_in)
        # else:
        #     part_f_out = final_feat.unsqueeze(-1)
        # if self.return_zero:
        #     part_f_out = torch.zeros_like(part_f_out)
        # return part_f_out.transpose(1,2), attn

    def word_attention(self, R, h, len_seq):
        """
        Input:
            R: hidden states of the entire words
            h: the final hidden state after processing the entire words
            len_seq: the length of the sequence
        Output:
            final_feat: the final feature after the bilinear attention
            attn: word attention weights
        """
        B, N, D = R.shape
        device = R.device
        len_seq = len_seq.to(device)

        W_attn = (self.w_attn * torch.eye(D).to(device))[None].repeat(B, 1, 1)
        score = torch.bmm(torch.bmm(R, W_attn), h.unsqueeze(-1))

        mask = torch.arange(N).reshape(1, N, 1).repeat(B, 1, 1).to(device)
        mask = mask < len_seq.reshape(B, 1, 1)

        score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, 1)
        final_feat = torch.bmm(R.transpose(1, 2), attn).squeeze(-1)

        return final_feat, attn.squeeze(-1)

@ENCODERS.register_module()
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, src_channel, tgt_channel, out_channel, qk_dim=512, nheads=8, qkv_bias=True, scale=True):
        super(MultiHeadCrossAttention, self).__init__()
        assert out_channel % nheads == 0
        feedforward_dim = out_channel * 2
        self.qk_ndim = qk_dim // nheads
        self.v_ndim = out_channel // nheads
        self.nheads=nheads
        self.Wq = nn.Linear(src_channel, qk_dim, bias=qkv_bias)
        self.Wk = nn.Linear(tgt_channel, qk_dim, bias=qkv_bias)
        self.Wv = nn.Linear(tgt_channel, out_channel, bias=qkv_bias)
        self.scale= 1 / math.sqrt(self.qk_ndim) if scale else 1. 

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)

        self.linear1 = nn.Linear(out_channel, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, out_channel)

        self.norm = nn.LayerNorm(out_channel)

    def forward(self, src, tgt, part_indicator):
        B, N, C = src.shape
        _, M, D = tgt.shape
        Q, K, V = self.Wq(src).reshape(B, N, self.nheads, self.qk_ndim), self.Wk(tgt).reshape(B, M, self.nheads, self.qk_ndim), self.Wv(tgt).reshape(B, M, self.nheads, self.v_ndim)
        
        QK = torch.matmul(Q.permute([0, 2, 1, 3]), K.permute([0, 2, 3, 1])) * self.scale # [B, head, N, M]
        QK = F.softmax(QK, dim=3)
        QK_dp = self.dp1(QK)
        QK_slice = QK_dp * part_indicator.reshape(B, 1, -1, 1)
        QK_slice = torch.sum(QK_slice, 2, keepdim=True)
        QK_dp = QK_slice
        message = torch.matmul(QK_dp, V.permute([0, 2, 1, 3])).transpose(1,2).reshape(B, -1, self.nheads * self.v_ndim)
        message2 = self.linear2(F.relu(self.linear1(message)))
        message = message + self.dp2(message2)
        message = self.norm(message)
        
        return message
    
@ENCODERS.register_module()
class LatentLanguageFuser(nn.Module):
    def __init__(
        self, 
        text_dim, 
        part_dim, 
        num_part, 
        residual=True, 
        cat_weights=True, 
        mid_dim=1024,
        conditional=False,
        conditional_dim=24,
        normalize_latent=False,
        regress_weights=False,
        direct_add_text_to_part=False,
        concat_weight=False):
        super().__init__()
        self.residual = residual
        self.normalize_latent=normalize_latent
        self.direct_add_text_to_part=direct_add_text_to_part
        if self.direct_add_text_to_part:
            assert not regress_weights
        self.concat_weight=concat_weight
        self.conditional=conditional
        self.num_part = num_part
        self.conditional_dim=conditional_dim if self.conditional else 0
        self.cat_weights=cat_weights
        self.part_latent_fuser = nn.ModuleList([nn.Linear(part_dim* num_part, part_dim),
                                                nn.Linear(part_dim + self.num_part if cat_weights else part_dim, part_dim),
                                                nn.Linear(part_dim + text_dim + self.conditional_dim, mid_dim),
                                                nn.Linear(mid_dim, mid_dim),
                                                nn.Linear(mid_dim, part_dim)]
                                                )
        self.regress_weights=regress_weights
        if self.regress_weights:
            self.weight_regressor = nn.Sequential(nn.Linear(text_dim, 128), nn.ReLU(inplace=True),
                                                  nn.Linear(128, 128), nn.ReLU(inplace=True),
                                                  nn.Linear(128, self.num_part), nn.Sigmoid())
        if self.direct_add_text_to_part:
            input_dim = text_dim
            if self.concat_weight:
                input_dim += self.num_part
            if self.conditional:
                input_dim += conditional_dim
            self.text_regressor = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(inplace=True),
                                                nn.Linear(128, 256), nn.ReLU(inplace=True),
                                                nn.Linear(256, 256), nn.ReLU(inplace=True),
                                                nn.Linear(256, part_dim))
         
    def forward(self, part_latent, part_indicators, text_latent, conditional=None):
        if self.conditional:
            assert conditional is not None and conditional.shape[-1] == self.conditional_dim
        part_id = part_indicators.max(1)[1].to(torch.int)
        B = part_latent.shape[0]
        if self.regress_weights:
            weights = self.weight_regressor(text_latent) #[B, num_part]
        else:
            weights = part_indicators
        if self.direct_add_text_to_part:
            input = torch.cat([text_latent, weights], dim=-1) if self.concat_weight else text_latent
            if self.conditional:
                K = conditional.shape[1]
                input = input.unsqueeze(1).repeat_interleave(K, dim=1)
            else:
                K = 1
                text_latent_out = self.text_regressor(input).unsqueeze(2).repeat_interleave(self.num_part, dim=2)
                
            out_latent = part_latent.unsqueeze(1) + weights.reshape(B, 1, self.num_part, 1) * text_latent_out.reshape(B, K, self.num_part, -1)
            if self.normalize_latent:
                out_latent = F.normalize(out_latent, dim=-1)
            return out_latent
                
        
            
        fused_latent = F.relu(self.part_latent_fuser[0](part_latent.reshape(B, -1)))
        
        if self.cat_weights:
            fused_latent = F.relu(self.part_latent_fuser[1](torch.cat([fused_latent, weights], dim=-1)))
        else:
            fused_latent = F.relu(self.part_latent_fuser[1](fused_latent))
        if self.conditional:
            K = conditional.shape[1]
            fused_latent = torch.cat([text_latent, fused_latent], dim=-1).unsqueeze(1).repeat_interleave(K, dim=1)
            fused_latent = torch.cat([fused_latent, conditional], dim=-1)
        else:
            K = 1
            fused_latent = torch.cat([text_latent, fused_latent], dim=-1).unsqueeze(1)
        
        fused_latent = F.relu(self.part_latent_fuser[2](fused_latent))
        fused_latent = self.part_latent_fuser[4](F.relu(self.part_latent_fuser[3](fused_latent)))
        if self.residual:
            fused_latent=part_latent.unsqueeze(1) + fused_latent
        out_latent = (1 - part_indicators).reshape(B, 1, self.num_part, 1) * part_latent.unsqueeze(1) + part_indicators.reshape(B, 1, self.num_part, 1) * fused_latent.unsqueeze(2)
        if self.normalize_latent:
            out_latent = F.normalize(out_latent, p=2, dim=-1)
        return out_latent
            
        
@ENCODERS.register_module()
class PartLanguageSelector(nn.Module):
    def __init__(self, text_dim, part_dim, out_channel):
        super(PartLanguageSelector, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(text_dim + part_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, out_channel)
        )

        self.dp = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, src, tgt, part_indicator):
        B, N, C = src.shape
        _, M, D = tgt.shape
        part_id = part_indicator.max(1)[1].to(torch.int)
        tgt_part = gather_operation(tgt.transpose(1,2).contiguous(), part_id.reshape(B, 1))
        return tgt_part.transpose(1,2)

        