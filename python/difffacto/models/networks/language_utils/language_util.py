import torch 
import pickle
def tokenizing(text: str):
    """
    Input:
        word2int: dict
        text: str
    Output:
        tensor of token: [len_seq]
    """
    word2int = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/python/anchor_diff/models/networks/language_utils/word2int.pkl", "rb"))

    token = list(map(lambda x: word2int[x], text.split(" ")))
    token = torch.tensor(token)

    return token