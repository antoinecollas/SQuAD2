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

class Translator():
    def __init__(self, vocabulary_size_in, vocabulary_size_out, max_seq=100, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        self.Transformer = Transformer(vocabulary_size_in, vocabulary_size_out, max_seq, nb_layers, nb_heads, d_model, nb_neurons, dropout)
        self.criterion = nn.CrossEntropyLoss()
        # print(list(self.Transformer.parameters()))
        self.optimizer = optim.Adam(self.Transformer.parameters(), lr=0.01, betas=(0.9,0.98), eps=1e-9)

    def train(self, X, Y):
        '''
        Arg:
            X: batch of phrases to translate: tensor(nb_texts, nb_tokens)
            Y: batch of translations: tensor(nb_texts, nb_tokens)
        '''
        batch_size = X.shape[0]
        translation = torch.zeros(batch_size, self.Transformer.max_seq).type(torch.LongTensor)
        for i in range(batch_size):
            translation[i][0] = BOS_IDX
        j=0
        running_loss = 0
        for i in range(1, self.Transformer.max_seq):
            output = self.Transformer(X, translation)
            output = output.contiguous().view(-1, output.size(-1))
            target = Y.contiguous().view(-1)
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            values, output = torch.max(output, dim=-1)
            while translation[0][j]!=0:
                j=j+1
            for i in range(batch_size):
                translation[i][j] = Y[i][j]
        print(running_loss/(self.Transformer.max_seq-1))
    
    def predict(self, X):
        '''
        Arg:
            X: phrases to translate: tensor(nb_texts, nb_tokens)
        '''
        nb_texts = X.size(0)
        targets = torch.zeros(nb_texts, self.Transformer.max_seq).type(torch.LongTensor)
        for i in range(nb_texts):
            targets[i][0] = BOS_IDX
        j=1
        running_loss = 0
        for i in range(1, self.Transformer.max_seq):
            output = self.Transformer(X, targets)
            # output = output[:,i]
            targets_temp = targets[:,i]
            self.optimizer.zero_grad()
            loss = self.criterion(output, targets_temp)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            # print(output)
            values, output = torch.max(output, dim=-1)
            for k in range(nb_texts):
                targets[k][j] = output[k]
            j=j+1
            # print(output[i])
            # print(targets[i])
        return targets
