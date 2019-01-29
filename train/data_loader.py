import torch, sys, collections, os
from sampler import *
from multiprocessing import Pool
from functools import partial

class DataLoader(object):
    def __init__(self, paths, constants, hyperparams, pad_Y_batch=True):
        with open(paths['bpe_source'], 'r') as f:
            bpe_src = f.readlines()
        with open(paths['bpe_target'], 'r') as f:
            bpe_tgt = f.readlines()

        for i in range(len(bpe_src)):
            bpe_src[i] = bpe_src[i].split(" ")
        for i in range(len(bpe_src)):
            bpe_tgt[i] = bpe_tgt[i].split(" ")

        self.constants = constants
        self.hyperparams = hyperparams
        self.current = 0
        self.bpe_src = bpe_src
        self.bpe_tgt = bpe_tgt
        self.batches_idx = list(SortishSampler(bpe_src, key=lambda x: len(bpe_src[x]), bs=self.hyperparams.BATCH_SIZE))
        self.pad_Y_batch = pad_Y_batch
        self.nb_texts = len(bpe_src)
        self.nb_batches = self.nb_texts//self.hyperparams.BATCH_SIZE
        if self.nb_texts % self.hyperparams.BATCH_SIZE != 0:
            self.nb_batches+=1
        if self.nb_batches<2:
            raise ValueError('There must be at least 2 batches.')
        
        self.set_itos()
        self.set_stoi_from_itos()

        #we remove phrases which are longer than MAX_SEQ (for memory and computation)
        self.rm_longest_phrases_tgt()
        
    def set_itos(self):
        phrases = self.bpe_src + self.bpe_tgt
        freq = collections.Counter(p for o in phrases for p in o)
        # print("Most common words:", freq.most_common(MAX_VOCAB))
        itos = [o for o,c in freq.most_common(self.hyperparams.MAX_VOCAB) if c>self.hyperparams.MIN_FREQ]
        #we add 4 constants
        for i in range(4):
            itos.insert(0, '')
        itos[self.constants.EOS_IDX] = self.constants.EOS_WORD
        itos[self.constants.BOS_IDX] = self.constants.BOS_WORD
        itos[self.constants.UNKNOW_WORD_IDX] = self.constants.UNKNOW_WORD
        itos[self.constants.PADDING_IDX] = self.constants.PADDING_WORD
        #we use a default value when the string doesn't exist in the dictionnary
        self.itos = itos

    def set_stoi_from_itos(self):
        res = {v:k for k,v in enumerate(self.itos)}
        self.stoi = collections.defaultdict(partial(int, self.constants.UNKNOW_WORD_IDX), res)
    
    def rm_longest_phrases_tgt(self):
        def longest(phrases):
            sorted_idx = list(SortSampler(phrases, key=lambda x: len(phrases[x])))
            to_remove = []
            for idx in sorted_idx:
                length = len(phrases[idx])
                if length <= self.hyperparams.MAX_SEQ:
                    break
                else:
                    to_remove.append(idx)
            return to_remove
        to_remove = longest(self.bpe_src)
        self.bpe_src = np.delete(self.bpe_src, to_remove)
        self.bpe_tgt = np.delete(self.bpe_tgt, to_remove)

    def pad_batch(self, batch, length=None):
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
                phrase.append(self.constants.PADDING_IDX)
            result[k] = np.array(phrase)
            k+=1
        return result
    
    def __iter__(self):
        self.current = 0
        return self

    def itotok(self, i):
        return [self.itos[int(o)] for o in i]

    def toktoi(self, tok):
        return [self.stoi[o] for o in tok]

    def __next__(self):
        if self.current >= self.nb_batches:
            raise StopIteration
        else:
            l = self.current
            src = self.bpe_src[self.batches_idx[l*self.hyperparams.BATCH_SIZE:(l+1)*self.hyperparams.BATCH_SIZE]]
            with Pool(self.constants.NCPUS) as p:
                src = np.array(p.map(self.toktoi, src))
            src = self.pad_batch(src)
            X_batch = torch.from_numpy(src).type(torch.LongTensor).to(self.constants.DEVICE)
            
            tgt = self.bpe_tgt[self.batches_idx[l*self.hyperparams.BATCH_SIZE:(l+1)*self.hyperparams.BATCH_SIZE]]
            if self.pad_Y_batch:
                with Pool(self.constants.NCPUS) as p:
                    tgt = np.array(p.map(self.toktoi, tgt))
                tgt = self.pad_batch(tgt)
                Y_batch = torch.from_numpy(tgt).type(torch.LongTensor).to(self.constants.DEVICE)
            else:
                # Y_batch = self.bpe_tgt[self.batches_idx[l*self.hyperparams.BATCH_SIZE:(l+1)*self.hyperparams.BATCH_SIZE]]
                raise NotImplementedError

            self.current += 1
            return X_batch, Y_batch
        
    def __len__(self):
        return len(self.bpe_src)