import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import MultiHeadAttention, PositionWiseFeedForward, get_mask

torch.manual_seed(1)
START_OF_SENTENCE_IDX = 1
END_OF_SENTENCE_IDX = 2

class Decoder(nn.Module):
    def __init__(self, n_layers=6, nb_heads=8, d_model=512, n_neurons = 2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.MultiHeadAttention = [MultiHeadAttention(nb_heads, d_model, dropout) for _ in range(self.n_layers)]
        self.EncoderDecoderAttention = [MultiHeadAttention(nb_heads, d_model, dropout) for _ in range(self.n_layers)]
        self.PositionWiseFeedForward = [PositionWiseFeedForward(d_model, n_neurons, dropout) for _ in range(self.n_layers)]

    def forward(self, X_encoder, X):
        '''
        Arg:
            X_encoder: output from encoder: tensor(nb_texts, nb_tokens, d_model(=size of one token))
            X: tensor(nb_texts, nb_tokens, d_model(=size of one token))
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        mask1 = get_mask(X,X)
        mask2 = get_mask(X,X_encoder)
        output = X
        for i in range(self.n_layers):
            output = self.MultiHeadAttention[i].forward(output,output,output,mask1)
            output = self.EncoderDecoderAttention[i].forward(output,X_encoder,X_encoder,mask2)
            output = self.PositionWiseFeedForward.[i]forward(output)
        return output

