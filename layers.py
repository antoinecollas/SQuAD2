import numpy as np
import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from constants import *

# torch.manual_seed(1)

def positionalEncoding(nb_words, nb_dimensions):
        X = torch.arange(0, nb_words).to(DEVICE, TYPE)
        Y = torch.arange(0, nb_dimensions).to(DEVICE, TYPE)
        Y, X = torch.meshgrid((Y, X))
        Y, X = Y.t(), X.t()
        TEMP = 10000
        temp1 = torch.sin(X/(torch.pow(TEMP, (2*Y)/nb_dimensions)))
        temp2 = torch.cos(X/(torch.pow(TEMP, (2*Y)/nb_dimensions)))
        Z = torch.zeros((nb_words, nb_dimensions)).to(DEVICE, TYPE)
        Z[:,0::2] = temp1[:,0::2]
        Z[:,1::2] = temp2[:,1::2]
        return Z

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
    
    def get_weights(self):
        return self.lookup_table.weight.data

def get_mask(X, Y, avoid_subsequent_info=False):
    '''
        Args:
        X: tensor(nb_texts, nb_tokens) it represents the initial sequence of tokens (before embedding)
        Y: tensor(nb_texts, nb_tokens) it represents the second initial sequence of tokens (before embedding)
        avoid_subsequent_info: bool: used for the first attention in decoding layer

        Output:
        tensor(nb_texts, nb_tokens, nb_tokens)
    '''
    # line = (X==PADDING_IDX).type(torch.FloatTensor)
    # line[line!=0] = float('-inf')
    # line = line.reshape(X.size(0), 1, X.size(1)).repeat(1, Y.size(1), 1).transpose(-2,-1)
    col = (Y==PADDING_IDX).to(DEVICE, TYPE)
    col[col!=0] = float('-inf')
    # print(col.shape)
    # print(X.shape)
    # print(Y.shape)
    col = col.reshape(Y.size(0), 1, Y.size(1)).repeat(1, X.size(1), 1)
    mask1 = col
    if avoid_subsequent_info:
        mask2_shape = (X.size(0), X.size(1), X.size(1))
        mask2 = torch.triu(torch.ones(mask2_shape[1:])).to(DEVICE, TYPE) - torch.eye(mask2_shape[1]).to(DEVICE, TYPE)
        mask2 = mask2.view(1, mask2_shape[1], mask2_shape[2]).expand(mask2_shape)
        mask2[mask2!=0] = float('-inf')
        mask1 = mask1+mask2
    return mask1.to(DEVICE, TYPE)

def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = K.shape[-1]
    if mask is not None:
        output = F.softmax((Q@(K.transpose(-2,-1))+mask)/np.sqrt(dk), dim=-1)
        #if a line contains only -inf (before softmax), then after softmax it contains only nan (because it does a division by 0)
        #we replace nan by 0
        # output[output!=output] = 0
        output = output@V
    else:
        output = F.softmax(Q@(K.transpose(-2,-1))/np.sqrt(dk), dim=-1)@V
    return output

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
        create_tensor = lambda size, std: nn.Parameter(nn.init.uniform_(torch.zeros(size, requires_grad=True), a=-std, b=std))
        self.W_q = create_tensor((h, self.d_model, self.d_k), stdv)
        self.W_k = create_tensor((h, self.d_model, self.d_k), stdv)
        self.W_v = create_tensor((h, self.d_model, self.d_v), stdv)
        stdv = 1. / math.sqrt(self.h*self.d_v)
        self.W_o = create_tensor((self.h*self.d_v, self.d_model), stdv)

    def forward(self, Q, K, V, mask=None):
        '''
            Args:
            Q: tensor(nb_texts, nb_tokens, d_model(=size of one token))
            K: tensor(nb_texts, nb_tokens, d_model(=size of one token))
            V: tensor(nb_texts, nb_tokens, d_model(=size of one token))

            Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        nb_texts, V_nb_tokens, d_model = V.shape
        V2 = V.repeat(1, self.h, 1).view(nb_texts, self.h, V_nb_tokens, d_model)
        nb_texts, Q_nb_tokens, d_model = Q.shape
        Q2 = Q.repeat(1, self.h, 1).view(nb_texts, self.h, Q_nb_tokens, d_model)
        nb_texts, K_nb_tokens, d_model = K.shape
        K2 = K.repeat(1, self.h, 1).view(nb_texts, self.h, K_nb_tokens, d_model)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1).view(nb_texts, self.h, Q_nb_tokens, K_nb_tokens)
            heads = scaled_dot_product_attention(Q2@self.W_q, K2@self.W_k, V2@self.W_v, mask)
        else:
            heads = scaled_dot_product_attention(Q2@self.W_q, K2@self.W_k, V2@self.W_v)
        heads = torch.cat(torch.unbind(heads, dim=1), dim=2) #concatenation
        output = heads@self.W_o
        output = self.dropout(output)
        output = output + Q
        output = self.layer_norm(output)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model = 512, nb_neurons = 2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        #He initialisation
        self.nonlinearity = 'relu'
        gain = nn.init.calculate_gain(self.nonlinearity)
        std = gain / math.sqrt(d_model)
        create_tensor = lambda size, std: nn.Parameter(nn.init.uniform_(torch.zeros(size, requires_grad=True), a=-std, b=std))
        self.W_1 = create_tensor((d_model, nb_neurons), std)
        self.b_1 = create_tensor(nb_neurons, std)
        std = 1 / math.sqrt(d_model)
        self.W_2 = create_tensor((nb_neurons, d_model), std)
        self.b_2 = create_tensor((d_model), std)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
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
        return self.layer_norm(self.dropout(self.dropout(self.activation(X@self.W_1+b_1))@self.W_2+b_2)+X)