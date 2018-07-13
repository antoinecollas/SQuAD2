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
    def __init__(self, vocabulary_size_in, vocabulary_size_out, nb_tokens_in, nb_tokens_out, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        self.Transformer = Transformer(vocabulary_size_in, vocabulary_size_out, nb_tokens_in, nb_tokens_out, nb_layers, nb_heads, d_model, nb_neurons, dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.Transformer.parameters(), lr=0.001, momentum=0.9)

    def train(self, X, Y, nb_epochs=50, batch_size=10):
        '''
        Arg:
            X: phrases to translate: tensor(nb_texts, nb_tokens)
            X: translations: tensor(nb_texts, nb_tokens)
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
            tensor(nb_texts, vocabulary_size)
            tensor(nb_texts, nb_tokens_out)
        '''
        nb_texts = X.size(0)
        nb_batches = nb_texts//batch_size
        for k in range(nb_epochs):
            print("Epoch:",k)
            for l in range(nb_batches):
                print("Batch:",l)
                X_batch = X[l*batch_size:(l+1)*batch_size]
                Y_batch = Y[l*batch_size:(l+1)*batch_size]
                targets = torch.zeros(batch_size, self.Transformer.nb_tokens_out).type(torch.LongTensor)
                for i in range(batch_size):
                    targets[i][0] = START_OF_SENTENCE_IDX
                j=0
                while targets[0][j]!=0:
                    j+=1
                for i in range(batch_size):
                    targets[i][j] = Y_batch[i][j]
                    # print(output[i])
                    # print(targets[i])
                for i in range(1, self.Transformer.nb_tokens_out):
                    output = self.Transformer(X_batch, targets)
                    output = output[:,i]
                    targets = targets[:,i]
                    self.optimizer.zero_grad()
                    print(output)
                    print(targets)
                    loss = self.criterion(output, targets)
                    loss.backward()
                    self.optimizer.step()
                    running_loss = loss.item()
                    print(running_loss)
                    values, output = torch.max(output, dim=-1)
        return phrases