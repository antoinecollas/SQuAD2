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
    def __init__(self, vocabulary_size_in, vocabulary_size_out, max_seq=100, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.max_seq = max_seq
        self.EmbeddingSrc = Embedding(vocabulary_size=vocabulary_size_in, d_model=d_model)
        self.EmbeddingTgt = Embedding(vocabulary_size=vocabulary_size_out, d_model=d_model)
        self.Encoder = Encoder(nb_layers=nb_layers, nb_heads=nb_heads, d_model=d_model, nb_neurons=nb_neurons, dropout=dropout)
        self.Decoder = Decoder(nb_layers=nb_layers, nb_heads=nb_heads, d_model=d_model, nb_neurons=nb_neurons, dropout=dropout)
        self.Linear = nn.Linear(d_model, vocabulary_size_out, bias=False)

    def forward(self, src, target):
        '''
        Arg:
            src: tensor(nb_texts, nb_tokens)
            target: tensor(nb_texts, nb_tokens)
        Output:
            tensor(nb_texts, nb_tokens, vocabulary_size_out)
        '''
        mask = get_mask(src,src)
        enc = self.EmbeddingSrc(src)
        enc = self.Encoder(enc,mask)
        mask1 = get_mask(target,target,avoid_subsequent_info=True)
        mask2 = get_mask(target,src)
        output = self.EmbeddingTgt(target)
        output = self.Decoder(enc, output, mask1, mask2)
        output = self.Linear(output)
        # output = F.softmax(output, dim=-1)
        return output