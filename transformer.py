import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import Embedding, get_mask
from encoder import Encoder
from decoder import Decoder
from constants import *

# torch.manual_seed(1)

class Transformer(nn.Module):
    def __init__(self, vocabulary_size_in, vocabulary_size_out, nb_tokens_in, nb_tokens_out, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.nb_tokens_in = nb_tokens_in
        self.nb_tokens_out = nb_tokens_out
        self.Embedding = Embedding(vocabulary_size_in, d_model)
        self.Encoder = Encoder(nb_layers, nb_heads, d_model, nb_neurons, dropout)
        self.Decoder = Decoder(nb_layers, nb_heads, d_model, nb_neurons, dropout)
        self.Linear = nn.Linear(nb_tokens_in*d_model, vocabulary_size_out, bias=False)

    def forward(self, src, target):
        '''
        Arg:
            src: tensor(nb_texts, nb_tokens)
            target: tensor(nb_texts, nb_tokens)
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
            tensor(nb_texts, vocabulary_size)
            tensor(nb_texts, nb_tokens_out)
        '''
        mask = get_mask(src,src)
        enc = self.Embedding(src)
        enc = self.Encoder(enc,mask)
        mask1 = get_mask(target,target,avoid_subsequent_info=True)
        mask2 = get_mask(target,src)
        output = self.Embedding(target)
        output = self.Decoder(enc, output, mask1, mask2)
        output = output.view(output.shape[0], -1)
        output = self.Linear(output)
        # output = F.softmax(output, dim=-1)
        return output