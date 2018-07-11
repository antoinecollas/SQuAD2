import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import MultiHeadAttention, PositionWiseFeedForward

torch.manual_seed(1)

class Encoder(nn.Module):
    def __init__(self, nb_heads=8, d_model=512, n_neurons = 2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(nb_heads, d_model, dropout)
        self.PositionWiseFeedForward = PositionWiseFeedForward(d_model, n_neurons, dropout)
        
    def forward(self, X):
        '''
        Arg:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        output = self.MultiHeadAttention.forward(X,X,X)
        return self.PositionWiseFeedForward.forward(output)

