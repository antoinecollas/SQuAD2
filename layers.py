import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def ScaledDotProductAttention(Q, K, V):
    dk = K.shape[-1]
    return F.softmax(Q@(K.transpose(-2,-1))/np.sqrt(dk), dim=-2)@V

class MultiHeadAttention(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = self.d_v = d_model//h
        self.weight = dict()
        self.weight["w_q"] = nn.Parameter(torch.Tensor(h, self.d_model, self.d_k))
        self.weight["w_k"] = nn.Parameter(torch.Tensor(h, self.d_model, self.d_k))
        self.weight["w_v"] = nn.Parameter(torch.Tensor(h, self.d_model, self.d_v))
        self.weight["w_o"] = nn.Parameter(torch.Tensor(self.h*self.d_v, self.d_model))
        self.reset_parameters()

    def reset_parameters(self):
        '''
            Initialize all the parameters. It follows the initialization of linear in pytorch:
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        '''
        for k, v in self.weight.items():
            stdv = 1. / math.sqrt(self.weight[k].size(1))
            v.data.uniform_(-stdv, stdv)

    def forward(self, Q, K, V):
        '''
            Args:
            Q: tensor(nb_texts, nb_tokens, d_model(=size of one word))
            K: tensor(nb_texts, nb_tokens, d_model(=size of one word))
            V: tensor(nb_texts, nb_tokens, d_model(=size of one word))

            Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one word))
        '''
        nb_texts, nb_tokens, d_model = Q.shape
        Q = Q.repeat(1, self.h, 1).view(nb_texts, self.h, nb_tokens, d_model)
        K = K.repeat(1, self.h, 1).view(nb_texts, self.h, nb_tokens, d_model)
        V = V.repeat(1, self.h, 1).view(nb_texts, self.h, nb_tokens, d_model)
        heads = ScaledDotProductAttention(Q@self.weight["w_q"], K@self.weight["w_k"], V@self.weight["w_v"])
        heads = torch.cat(torch.unbind(heads, dim=1), dim=2) #concatenation
        return heads@self.weight["w_o"]