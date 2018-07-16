import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import MultiHeadAttention, PositionWiseFeedForward, get_mask

# torch.manual_seed(1)

class Decoder(nn.Module):
    def __init__(self, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.nb_layers = nb_layers
        self.MultiHeadAttention = nn.ModuleList([MultiHeadAttention(nb_heads, d_model, dropout) for _ in range(self.nb_layers)])
        self.EncoderDecoderAttention = nn.ModuleList([MultiHeadAttention(nb_heads, d_model, dropout) for _ in range(self.nb_layers)])
        self.PositionWiseFeedForward = nn.ModuleList([PositionWiseFeedForward(d_model, nb_neurons, dropout) for _ in range(self.nb_layers)])

    def forward(self, X_encoder, X, mask1=torch.Tensor(), mask2=torch.Tensor()):
        '''
        Arg:
            X_encoder: output from encoder: tensor(nb_texts, nb_tokens, d_model(=size of one token))
            X: tensor(nb_texts, nb_tokens, d_model(=size of one token))
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        output = X
        for i in range(self.nb_layers):
            output = self.MultiHeadAttention[i].forward(output,output,output,mask1)
            output = self.EncoderDecoderAttention[i].forward(output,X_encoder,X_encoder,mask2)
            output = self.PositionWiseFeedForward[i].forward(output)

        return output