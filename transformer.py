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
    def __init__(self, vocabulary_size_in, vocabulary_size_out, nb_tokens_in, nb_tokens_out, n_layers=6, nb_heads=8, d_model=512, n_neurons = 2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.nb_tokens_in = nb_tokens_in
        self.nb_tokens_out = nb_tokens_out
        self.Embedding = Embedding(vocabulary_size_in, d_model)
        self.Encoder = Encoder(n_layers, nb_heads, d_model, n_neurons, dropout)
        self.Decoder = Decoder(n_layers, nb_heads, d_model, n_neurons, dropout)
        self.Linear = nn.Linear(d_model, vocabulary_size_out, bias=False)

    def forward(self, X):
        '''
        Arg:
            X: tensor(nb_texts, nb_tokens)
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
            tensor(nb_texts, vocabulary_size)
            tensor(nb_texts, nb_tokens_out)
        '''
        mask = get_mask(X,X)
        enc = self.Embedding(X)
        enc = self.Encoder(enc,mask)
        nb_texts = X.size(0)
        phrases = torch.zeros(nb_texts, self.nb_tokens_out).type(torch.LongTensor)
        for i in range(nb_texts):
            phrases[i][0] = START_OF_SENTENCE_IDX
        for i in range(1, self.nb_tokens_out):
            print("itération n°", i)
            print(phrases)
            sequence_for_mask = torch.zeros(nb_texts, self.nb_tokens_out).type(torch.LongTensor)
            sequence_for_mask[:,:i] = START_OF_SENTENCE_IDX
            mask1 = get_mask(sequence_for_mask,sequence_for_mask,avoid_subsequent_info=True)
            mask2 = get_mask(sequence_for_mask,X)
            output = self.Embedding(phrases)
            output = self.Decoder(enc, output, mask1, mask2)
            output = self.Linear(output)
            output = F.softmax(output, dim=-1)
            values, output = torch.max(output, dim=-1)
            for j in range(nb_texts):
                phrases[j][i] = output[j][i]

        return phrases