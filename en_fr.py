from translator import *
import sys
sys.path.append("../fastai/")
from fastai.text import *
from constants import *
from multiprocessing import Pool
import os
from functools import partial
import time

'''
    Translation from English to French
'''


def toktoi(stoi, tok):
    return [stoi[o] for o in tok]

class Toktoi(object):
    def __init__(self, stoi):
        self.stoi = stoi
    def __call__(self, tok):
        return toktoi(self.stoi, tok)

def itotok(itos, i):
    return [itos[o] for o in i]

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

def prepare_texts(filename, lang):
    with open(filename,"r") as f:
        phrases = f.readlines()
    t0 = time.time()
    tok = Tokenizer(lang=lang).proc_all_mp(partition(phrases, len(phrases)//NCPUS + 1), ncpus=NCPUS)
    t1 = time.time()
    total = t1-t0
    print("time Tokenizer=",total)
    t0 = time.time()
    freq = Counter(p for o in tok for p in o)
    # print("Most common words:", freq.most_common(25))
    itos = [o for o,c in freq.most_common(MAX_VOCAB) if c>MIN_FREQ]
    itos.insert(0, EOS_WORD)
    itos.insert(0, BOS_WORD)
    itos.insert(0, UNKNOW_WORD)
    itos.insert(0, PADDING_WORD)
    # print("Length dictionnary integer to string=", len(itos))
    #we use a default value when the string doesn't exist in the dictionnary
    stoi = stoi_from_itos(itos)
    t1 = time.time()
    total = t1-t0
    print("time dictionaries=",total)
    # toktoi = lambda p: [stoi[o] for o in p]
    # texts = np.array([toktoi(p) for p in tok])
    return tok, itos, stoi

print("====ENGLISH PHRASES====")
tok_en, itos_en, stoi_en = prepare_texts(ENGLISH_FILENAME, lang='en')
t0 = time.time()
texts_en = np.array(Pool(NCPUS).map(Toktoi(stoi_en), tok_en))
t1 = time.time()
total = t1-t0
print("time tok to int=",total)

print("====FRENCH PHRASES====")
tok_fr, itos_fr, stoi_fr = prepare_texts(FRENCH_FILENAME, lang='fr')
t0 = time.time()
texts_fr = np.array(Pool(NCPUS).map(Toktoi(stoi_fr), tok_fr))
t1 = time.time()
total = t1-t0
print("time tok to int=",total)

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

batches_idx = list(SortishSampler(texts_en, key=lambda x: len(texts_en[x]), bs=BATCH_SIZE))

nb_texts = len(texts_en)
nb_batches = nb_texts//BATCH_SIZE

tr = Translator(vocabulary_size_in=len(stoi_en),vocabulary_size_out=len(stoi_fr),max_seq=MAX_SEQ,nb_layers=NB_LAYERS,nb_heads=NB_HEADS,d_model=D_MODEL,nb_neurons=NB_NEURONS)
tr.train(True)
tr.to(DEVICE)

if not PRETRAIN:
    print("=======TRAINING=======")
    for k in range(NB_EPOCH):
        print("=======Epoch:=======",k)
        for l in range(nb_batches):
            print("Batch:",l)
            X_batch = torch.from_numpy(pad_batch(texts_en[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
            Y_batch = torch.from_numpy(pad_batch(texts_fr[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]], length=MAX_SEQ)).type(torch.LongTensor).to(DEVICE)
            tr.fit(X_batch, Y_batch)
    torch.save(tr.state_dict(), PATH_WEIGHTS)
else:
    tr = Translator(vocabulary_size_in=len(stoi_en),vocabulary_size_out=len(stoi_fr),max_seq=MAX_SEQ,nb_layers=NB_LAYERS,nb_heads=NB_HEADS,d_model=D_MODEL,nb_neurons=NB_NEURONS)
    tr.load_state_dict(torch.load(PATH_WEIGHTS))
    tr.to(DEVICE)

print("=======PREDICTION=======")
l=0
X_batch = torch.from_numpy(pad_batch(texts_en[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
Y_batch = torch.from_numpy(pad_batch(texts_fr[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]], length=MAX_SEQ)).type(torch.LongTensor).to(DEVICE)
prediction = tr.predict(X_batch).to(torch.device("cpu"))
translations = np.array(Pool(NCPUS).map(Itotok(itos_fr), list(prediction)))
X_batch = np.array(Pool(NCPUS).map(Itotok(itos_en), list(X_batch.to(torch.device("cpu")))))
Y_batch = np.array(Pool(NCPUS).map(Itotok(itos_fr), list(Y_batch.to(torch.device("cpu")))))
print(X_batch[:10,:10])
print(translations[:10,:10])
print("=======ORIGINAL=======")
print(Y_batch[:10,:10])