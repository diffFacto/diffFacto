import torch 
from torch import nn 
import torch.nn.functional as F
import math

class SinusoidalEmbedding3D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim=out_dim 
        self.intermediate_dim = (out_dim // 6) * 6

        self.scaling_factor = torch.ones(self.intermediate_dim // 6, dtype=torch.float32) * (10000 ** (1 / self.intermediate_dim))
        self.scaling_factor = torch.pow(self.scaling_factor, torch.arange(self.intermediate_dim // 6) * 6)

    def forward(self, x):
        # x[B, N, 3]
        B, N, _ = x.shape
        out = torch.zeros([B, N, self.out_dim]).cuda()
        cos_x, sin_x = torch.cos(x.unsqueeze(3) / self.scaling_factor.reshape(1, 1, 1, -1).to(x.device)), torch.sin(x.unsqueeze(3) / self.scaling_factor.reshape(1, 1, 1, -1).to(x.device)) #[B, N, 3, D // 6]
        sinusoidal_pe = torch.stack([sin_x, cos_x], dim=-1).reshape(B, N, self.intermediate_dim) #[B, N, 3, D // 6, 2] -> [B, N, D]
        out[:, :, :self.intermediate_dim] += sinusoidal_pe
        return out
        




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel, qk_dim=512, nheads=8, qkv_bias=True, scale=True):
        super(MultiHeadSelfAttention, self).__init__()
        assert out_channel % nheads == 0
        self.ndim = qk_dim // nheads
        self.nheads=nheads
        self.out_channel=out_channel
        self.Wq = nn.Linear(in_channel, qk_dim, bias=qkv_bias)
        self.Wk = nn.Linear(in_channel, qk_dim, bias=qkv_bias)
        self.Wv = nn.Linear(in_channel, out_channel, bias=qkv_bias)
        self.scale= 1 / math.sqrt(self.ndim) if scale else 1. 

    def forward(self, x):
        ori_x = x
        B, N, C = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        V = V.reshape(B, N, self.nheads, self.out_channel // self.nheads)
        Q, K = map(lambda w: w.reshape(B, N, self.nheads, self.ndim), [Q, K])
        QK = torch.matmul(Q.permute([0, 2, 1, 3]), K.permute([0, 2, 3, 1])) * self.scale # [B, head, N, N]
        QK = F.softmax(QK, dim=3)
        message = torch.matmul(QK, V.permute([0, 2, 1, 3])).transpose(1,2).reshape(B, N, self.out_channel)
        return message


class GAT(nn.Module):
    def __init__(self, in_channel, out_channel, nheads, dropout=0., alpha=0.2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        assert out_channel % nheads == 0
        nhid = out_channel // nheads

        self.attentions = [GraphAttentionLayer(in_channel, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x):
        ori_x = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = torch.einsum("bni, ij->bnj", h, self.W) # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        attention = self._prepare_attentional_mechanism_input(Wh) # [B, N, N]

        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum("bij, bjn->bin", attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)
        Wh1 = torch.einsum("bni, ij->bnj", Wh, self.a[:self.out_features, :])
        Wh2 = torch.einsum("bni, ij->bnj", Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1,2)
        return self.leakyrelu(e)


if __name__ == '__main__':
    pe_embedding = SinusoidalEmbedding3D(6)
    x = torch.tensor([[1,1,1], [0,1,0], [0,0,1]],dtype=torch.float32).reshape(1, 3, 3)
    embed = pe_embedding(x)
    print(embed)
    print(embed.shape)
