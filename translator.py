import numpy as np
import math, torch, time
# from apex import amp
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer import Transformer
from constants import *

# torch.manual_seed(1)

class CustomOptimizer():
    def __init__(self, parameters, d_model, warmup_steps=4000, betas=(0.9,0.98), eps=1e-9):
        self.opt = optim.Adam(parameters, betas=betas, eps=eps)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 1

    def set_next_lr(self):
        # self.lr = (self.d_model**(-0.5) * min(self.step_num**(-0.5), self.step_num*(self.warmup_steps**(-1.5))))/10
        self.lr = self.d_model**(-0.5) * min(self.step_num**(-0.5), self.step_num*(self.warmup_steps**(-1.5)))
        for p in self.opt.param_groups:
            p['lr'] = 0.001
        # print("learning rate=", self.lr)
        self.step_num = self.step_num + 1
    
    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.lr = self.set_next_lr()
        self.opt.step()

class Translator(nn.Module):
    def __init__(self, vocabulary_size_in, vocabulary_size_out, share_weights=True, max_seq=100, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1, warmup_steps=4000):
        super(Translator, self).__init__()
        self.Transformer = Transformer(vocabulary_size_in, vocabulary_size_out, share_weights, max_seq, nb_layers, nb_heads, d_model, nb_neurons, dropout)
        self.criterion = nn.CrossEntropyLoss()
        # print(list(self.Transformer.parameters()))
        self.optimizer = CustomOptimizer(self.Transformer.parameters(), d_model=d_model, warmup_steps=warmup_steps)
        # self.amp_handle = amp.init()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        # with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        self.optimizer.step()
        running_loss = loss.item()
        return running_loss

    def predict(self, X):
        '''
        Arg:
            X: batch of phrases to translate: tensor(nb_texts, nb_tokens)
        '''
        self.train(False)
        batch_size, max_seq = X.shape
        max_seq += 10 #TODO: remove hard code
        temp = torch.zeros(batch_size, max_seq).type(torch.LongTensor).to(DEVICE)
        temp[:,0] = BOS_IDX
        enc = self.Transformer.forward_encoder(X)
        for j in range(1, max_seq):
            # print(j)
            output = self.Transformer.forward_decoder(X, enc, temp)
            output = torch.argmax(output, dim=-1)
            temp[:,j] = output[:,j-1]
        #remove padding
        translations = []
        for translation in temp:
            temp2 = []
            for i in range(max_seq):
                if translation[i] == PADDING_IDX:
                    break
                if i!=0:
                    temp2.append(translation[i])
            translations.append(temp2)
        return translations