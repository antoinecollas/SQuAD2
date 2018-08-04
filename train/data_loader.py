import torch
from constants import *
from sampler import *

class DataLoader(object):
    def __init__(self, texts_src, texts_tgt, batch_size, device, pad_Y_batch=True):
        self.current = 0
        self.texts_src = texts_src
        self.texts_tgt = texts_tgt
        self.batch_size = batch_size
        self.batches_idx = list(SortishSampler(texts_src, key=lambda x: len(texts_src[x]), bs=self.batch_size))
        self.device = device
        self.pad_Y_batch = pad_Y_batch
        self.nb_texts = len(texts_src)
        self.nb_batches = self.nb_texts//self.batch_size
        if self.nb_texts % self.batch_size != 0:
            self.nb_batches+=1
        if self.nb_batches<2:
            raise ValueError('There must be at least 2 batches.')

    @staticmethod
    def pad_batch(batch, length=None):
        if length is None:
            len_max = -float('Inf')
            for phrase in batch:
                if len(phrase)>len_max:
                    len_max = len(phrase)
        else:
            len_max=length
        result = np.zeros((batch.shape[0], len_max))
        k=0
        for phrase in batch:
            for i in range(len(phrase), len_max):
                phrase.append(PADDING_IDX)
            result[k] = np.array(phrase)
            k+=1
        return result
    
    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= self.nb_batches:
            raise StopIteration
        else:
            l = self.current
            X_batch = torch.from_numpy(self.pad_batch(self.texts_src[self.batches_idx[l*self.batch_size:(l+1)*self.batch_size]])).type(torch.LongTensor).to(self.device)
            if self.pad_Y_batch:
                Y_batch = torch.from_numpy(self.pad_batch(self.texts_tgt[self.batches_idx[l*self.batch_size:(l+1)*self.batch_size]])).type(torch.LongTensor).to(self.device)
            else:
                Y_batch = self.texts_tgt[self.batches_idx[l*self.batch_size:(l+1)*self.batch_size]]
            self.current += 1
            return X_batch, Y_batch

