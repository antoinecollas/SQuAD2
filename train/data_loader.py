import torch
from constants import *
from sampler import *
from multiprocessing import Pool

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

def toktoi(stoi, tok):
    return [stoi[o] for o in tok]

class Toktoi(object):
    def __init__(self, stoi):
        self.stoi = stoi
    def __call__(self, tok):
        return toktoi(self.stoi, tok)

def itotok(itos, i):
    return [itos[int(o)] for o in i]

class Itotok(object):
    def __init__(self, itos):
        self.itos = itos
    def __call__(self, i):
        return itotok(self.itos, i)

def unknow_word():
    return UNKNOW_WORD_IDX

def stoi_from_itos(itos):
    res = {v:k for k,v in enumerate(itos)}
    return collections.defaultdict(unknow_word, res)

def tokenize(filename_bpe_source, filename_bpe_target):
    t0 = time.time()
    with open(filename_bpe_source,"r") as f:
        phrases_source = f.readlines()
    with open(filename_bpe_target,"r") as f:
        phrases_target = f.readlines()
    for i, phrase in enumerate(phrases_source):
        phrases_source[i] = phrase.split(" ")
    for i, phrase in enumerate(phrases_target):
        phrases_target[i] = phrase.split(" ")
    t1 = time.time()
    total = t1-t0
    print("time tokenize (split)=",total)
    return phrases_source, phrases_target

def get_itos_stoi(phrases_source, phrases_target):
    phrases = phrases_source + phrases_target
    t0 = time.time()
    freq = collections.Counter(p for o in phrases for p in o)
    # print("Most common words:", freq.most_common(MAX_VOCAB))
    itos = [o for o,c in freq.most_common(MAX_VOCAB) if c>MIN_FREQ]
    #we add the 4 constants
    for i in range(4):
        itos.insert(0, '')
    itos[EOS_IDX] = EOS_WORD
    itos[BOS_IDX] = BOS_WORD
    itos[UNKNOW_WORD_IDX] = UNKNOW_WORD
    itos[PADDING_IDX] = PADDING_WORD
    print("Length dictionnary integer to string=", len(itos))
    #we use a default value when the string doesn't exist in the dictionnary
    stoi = stoi_from_itos(itos)
    t1 = time.time()
    total = t1-t0
    return itos, stoi

# print("====STOI AND ITOS====")
# train_tok_en, train_tok_fr = tokenize(paths['bpe_source_train'], paths['bpe_target_train'])
# itos, stoi = get_itos_stoi(train_tok_en, train_tok_fr)
# with open(os.path.join(folder, PREPROCESSED_STOI_FILE), 'wb') as f:
#     pickle.dump(dict(stoi.items()), f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(os.path.join(folder, PREPROCESSED_ITOS_FILE), 'wb') as f:
#     pickle.dump(itos, f, protocol=pickle.HIGHEST_PROTOCOL)

# test_tok_source, test_tok_target = tokenize(paths['bpe_source_test'], paths['bpe_target_test'])

# print("====TOK TO INT SOURCE====")
# t0 = time.time()
# train_texts_en = np.array(Pool(NCPUS).map(Toktoi(stoi), train_tok_en))
# test_texts_en = np.array(Pool(NCPUS).map(Toktoi(stoi), test_tok_source))
# t1 = time.time()
# total = t1-t0
# print("time tok to int=",total)

# print("====TOK TO INT TARGET====")
# t0 = time.time()
# train_texts_target = np.array(Pool(NCPUS).map(Toktoi(stoi), train_tok_fr))
# test_texts_target = np.array(Pool(NCPUS).map(Toktoi(stoi), test_tok_target))
# t1 = time.time()
# total = t1-t0
# print("time tok to int=",total)