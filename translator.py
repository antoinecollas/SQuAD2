import numpy as np
import math, torch, time
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer import Transformer
from constants import *
import matplotlib.pyplot as plt

# torch.manual_seed(1)

class PaperScheduler():
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 1

    def plot_lr(self, nb_train_steps):
        lrs = [self._lr(self.d_model, step_num, self.warmup_steps) for step_num in range(1, nb_train_steps)]
        plt.plot(range(1,nb_train_steps), lrs)
        plt.xlabel('Training step')
        plt.ylabel('learning rate')
        plt.title('Learning rate for ' + str(nb_train_steps) + ' training steps.')
        plt.show()

    @staticmethod
    def _lr(d_model, step_num, warmup_steps):
        #paper
        # return d_model**(-0.5) * min(step_num**(-0.5), step_num*(warmup_steps**(-1.5)))
        #tensor2tensor
        return 2 * min(1, step_num/warmup_steps) * (1/math.sqrt(max(step_num, warmup_steps))) * (1/math.sqrt(d_model))

    def set_next_lr(self):
        self.lr = self._lr(self.d_model, self.step_num, self.warmup_steps)
        for p in self.opt.param_groups:
            p['lr'] = self.lr
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
        self.optimizer = optim.Adam(self.Transformer.parameters(), betas=(0.9, 0.98), eps=1e-9) #TODO remove hardcode
        self.scheduler = PaperScheduler(self.optimizer, d_model=d_model, warmup_steps=warmup_steps)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(self, data_iter, nb_epoch, verbose=True):
        '''
        Arg:
            data_iter: iterator which gives two batches: one of source language and one for target language
            nb_epoch: int
        '''
        self.training_loss_tab = []
        for k in range(nb_epoch):
            print("=======Epoch:=======",k)
            training_loss = 0
            for i, (X, Y) in enumerate(data_iter):
                batch_size = X.shape[0]
                bos = torch.zeros(batch_size, 1).fill_(BOS_IDX).type(torch.LongTensor).to(DEVICE)
                translation = torch.cat((bos, Y[:,:-1]),dim=1)
                output = self.Transformer(X, translation)
                output = output.contiguous().view(-1, output.size(-1))
                target = Y.contiguous().view(-1)
                self.scheduler.zero_grad()
                loss = self.criterion(output, target)
                training_loss += + loss.item()
                loss.backward()
                self.scheduler.step()
                if i==(data_iter.nb_batches-1):
                    training_loss = training_loss/data_iter.nb_batches
                    self.training_loss_tab.append(float(training_loss))
            if k%50==0: #TODO: remove hardcode
                torch.save(self.state_dict(), PATH_WEIGHTS)
            if verbose:
                print(float(training_loss))
        return training_loss

    def plot_training_loss(self):
        plt.plot(range(1,len(self.training_loss_tab)+1), self.training_loss_tab)
        plt.xlabel('Epoch')
        plt.ylabel('Training loss')
        plt.show()

    def translate(self, X):
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