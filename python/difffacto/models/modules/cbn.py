import torch
import torch.nn as nn
from difffacto.utils.misc import timestep_embedding


'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''
class ConditionalBatchNorm1d(nn.Module):

    def __init__(self, channels, embed_t_size=128, emb_size=256, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(ConditionalBatchNorm1d, self).__init__()

        self.embed_t_size = embed_t_size # size of the lstm emb which is input to MLP
        self.emb_size = emb_size # size of hidden layer of MLP
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.channels = channels

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.channels))
        self.gammas = nn.Parameter(torch.ones(self.channels))
        self.betas.requires_grad = False
        self.gammas.requires_grad = False
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
          nn.Linear(self.embed_t_size, self.emb_size),
          nn.ReLU(inplace=True),
          nn.Linear(self.emb_size, self.channels),
        )

        self.fc_beta = nn.Sequential(
          nn.Linear(self.embed_t_size, self.emb_size),
          nn.ReLU(inplace=True),
          nn.Linear(self.emb_size, self.channels),
        )

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel
    Arguments:
        lstm_emb : lstm embedding of the question
    Returns:
        delta_betas, delta_gammas : for each layer
    '''
    def create_cbn_input(self, t):
        t_embed = timestep_embedding(t, self.embed_t_size)
        if self.use_betas:
            delta_betas = self.fc_beta(t_embed)
        else:
            delta_betas = torch.zeros(self.channels).to(t.device)

        if self.use_gammas:
            delta_gammas = self.fc_gamma(t_embed)
        else:
            delta_gammas = torch.zeros(self.channels).to(t.device)

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values
    Arguments:
        feature : feature map from the previous layer
        lstm_emb : lstm embedding of the question
    Returns:
        out : beta and gamma normalized feature map
        lstm_emb : lstm embedding of the question (unchanged)
    Note : lstm_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require lstm question embeddings
    '''
    def forward(self, feature, t):
        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(t)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature, dim=(0,2), keepdim=True)
        batch_var = torch.var(feature, dim=(0,2), keepdim=True)

        # extend the betas and gammas of each channel across the height and width of feature map

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = feature_normalized, gammas_cloned.unsqueeze(2) + betas_cloned.unsqueeze(2)

        return out