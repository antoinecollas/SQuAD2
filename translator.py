import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer import Transformer
from constants import *

# torch.manual_seed(1)

class Translator(nn.Module):
    def __init__(self, vocabulary_size_in, vocabulary_size_out, max_seq=100, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        super(Translator, self).__init__()
        self.Transformer = Transformer(vocabulary_size_in, vocabulary_size_out, max_seq, nb_layers, nb_heads, d_model, nb_neurons, dropout)
        self.criterion = nn.CrossEntropyLoss()
        # print(list(self.Transformer.parameters()))
        self.optimizer = optim.Adam(self.Transformer.parameters(), lr=0.01, betas=(0.9,0.98), eps=1e-8)

    # def to(self, device):
    #     self.Transformer.to(device)

    # def train(self, mode):
    #     self.Transformer.train(mode)

    def fit(self, X, Y):
        '''
        Arg:
            X: batch of phrases to translate: tensor(nb_texts, nb_tokens)
            Y: batch of translations: tensor(nb_texts, nb_tokens)
        '''
        batch_size = X.shape[0]
        bos = torch.zeros(batch_size, 1).fill_(BOS_IDX).type(torch.LongTensor).to(DEVICE)
        translation = torch.cat((bos, Y[:,:-1]),dim=1)
        output = self.Transformer(X, translation)
        output = output.contiguous().view(-1, output.size(-1))
        target = Y.contiguous().view(-1)
        self.optimizer.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        running_loss = loss.item()
        print(running_loss)

    def predict(self, X):
        '''
        Arg:
            X: batch of phrases to translate: tensor(nb_texts, nb_tokens)
        '''
        self.train(False)
        batch_size = X.shape[0]
        translation = torch.zeros(batch_size, self.Transformer.max_seq).type(torch.LongTensor).to(DEVICE)
        translation[:,0] = BOS_IDX
        j=1
        running_loss = 0
        for _ in range(1, self.Transformer.max_seq):
            output = self.Transformer(X, translation)
            output = torch.argmax(output, dim=-1)
            for i in range(batch_size):
                translation[i][j] = output[i][j-1]
            j=j+1
        return translation[:,1:]
