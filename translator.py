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
        self.optimizer = optim.Adam(self.Transformer.parameters(), lr=0.001, betas=(0.9,0.98), eps=1e-8)

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
        translation = torch.zeros(batch_size, self.Transformer.max_seq).type(torch.LongTensor).to(DEVICE)
        for i in range(batch_size):
            translation[i][0] = BOS_IDX
        j=1
        running_loss = 0
        for i in range(1, self.Transformer.max_seq):
            # print("i=",i)
            # print("X=", X[0])
            # print("Y=", Y[0])
            # print("translation=", translation[0])
            output = self.Transformer(X, translation)
            output = output.contiguous().view(-1, output.size(-1))
            target = Y.contiguous().clone()
            target[:,i:] = 0
            target = target.view(-1)
            self.optimizer.zero_grad()
            # print("output=", output[1])
            # print("target[:10]=", target[:10])
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            # values, output = torch.max(output, dim=-1)
            # while translation[0][j]!=0:
            for k in range(batch_size):
                translation[k][j] = Y[k][j-1]
            j=j+1
        print(running_loss/(self.Transformer.max_seq-1))
    
    def predict(self, X):
        '''
        Arg:
            X: batch of phrases to translate: tensor(nb_texts, nb_tokens)
        '''
        self.train(False)
        batch_size = X.shape[0]
        translation = torch.zeros(batch_size, self.Transformer.max_seq).type(torch.LongTensor).to(DEVICE)
        for i in range(batch_size):
            translation[i][0] = BOS_IDX
        j=1
        running_loss = 0
        for _ in range(1, self.Transformer.max_seq):
            output = self.Transformer(X, translation)
            output = torch.argmax(output, dim=-1)
            for i in range(batch_size):
                # print("output[i]=", output[i])
                translation[i][j] = output[i][j-1]
            j=j+1
            # print("translation.shape=", translation.shape)
        return translation
