import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
PADDING_IDX = 0

def positionalEncoding(nb_words, nb_dimensions):
        X = np.arange(0, nb_words)
        Y = np.arange(0, nb_dimensions)
        Y, X = np.meshgrid(Y, X)
        TEMP = 10000
        temp1 = np.sin(X/(np.power(TEMP, (2*Y)/nb_dimensions)))
        temp2 = np.cos(X/(np.power(TEMP, (2*Y)/nb_dimensions)))
        Z = np.zeros((nb_words, nb_dimensions))
        Z[:,0::2] = temp1[:,0::2]
        Z[:,1::2] = temp2[:,1::2]
        return torch.from_numpy(Z).type(torch.FloatTensor)

class Embedding(nn.Module):
    def __init__(self, vocabulary_size, d_model=512):
        super(Embedding, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.lookup_table = nn.Embedding(self.vocabulary_size, self.d_model, padding_idx=PADDING_IDX)
    
    def forward(self, X):
        '''
            Args:
            X: tensor(nb_texts, nb_tokens)

            Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        vectors = self.lookup_table(X)
        return vectors + positionalEncoding(vectors.size(1), self.d_model)

def ScaledDotProductAttention(Q, K, V):
    dk = K.shape[-1]
    return F.softmax(Q@(K.transpose(-2,-1))/np.sqrt(dk), dim=-2)@V

class MultiHeadAttention(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = self.d_v = d_model//h
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

        #Initialize all the parameters. It follows the initialization of linear in pytorch:
        #https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        stdv = 1. / math.sqrt(self.d_model)
        self.W_q = nn.Parameter(torch.Tensor(h, self.d_model, self.d_k)).data.normal_(0, stdv)
        self.W_k = nn.Parameter(torch.Tensor(h, self.d_model, self.d_k)).data.normal_(0, stdv)
        self.W_v = nn.Parameter(torch.Tensor(h, self.d_model, self.d_v)).data.normal_(0, stdv)
        stdv = 1. / math.sqrt(self.h*self.d_v)
        self.W_o = nn.Parameter(torch.Tensor(self.h*self.d_v, self.d_model)).data.normal_(0, stdv)

    def forward(self, Q, K, V):
        '''
            Args:
            Q: tensor(nb_texts, nb_tokens, d_model(=size of one token))
            K: tensor(nb_texts, nb_tokens, d_model(=size of one token))
            V: tensor(nb_texts, nb_tokens, d_model(=size of one token))

            Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        nb_texts, nb_tokens, d_model = Q.shape
        Q2 = Q.repeat(1, self.h, 1).view(nb_texts, self.h, nb_tokens, d_model)
        K2 = K.repeat(1, self.h, 1).view(nb_texts, self.h, nb_tokens, d_model)
        V2 = V.repeat(1, self.h, 1).view(nb_texts, self.h, nb_tokens, d_model)
        heads = ScaledDotProductAttention(Q2@self.W_q, K2@self.W_k, V2@self.W_v)
        heads = torch.cat(torch.unbind(heads, dim=1), dim=2) #concatenation
        output = heads@self.W_o
        output = self.dropout(output)
        output = output + Q
        output = self.layer_norm(output)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model = 512, n_neurons = 2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        #He initialisation
        self.nonlinearity = 'relu'
        gain = nn.init.calculate_gain(self.nonlinearity)
        std = gain / math.sqrt(d_model)
        self.W_1 = nn.Parameter(torch.Tensor(d_model, n_neurons)).data.normal_(0, std)
        self.b_1 = nn.Parameter(torch.Tensor(n_neurons)).data.normal_(0, std)
        std = 1 / math.sqrt(d_model)
        self.W_2 = nn.Parameter(torch.Tensor(n_neurons, d_model)).data.normal_(0, std)
        self.b_2 = nn.Parameter(torch.Tensor(d_model)).data.normal_(0, std)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        '''
        Arg:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        nb_texts, nb_tokens, d_model = X.shape
        b_1 = self.b_1.repeat(nb_texts, nb_tokens, 1)
        b_2 = self.b_2.repeat(nb_texts, nb_tokens, 1)
        return self.layer_norm(self.dropout(self.activation(X@self.W_1+b_1)@self.W_2+b_2)+X)