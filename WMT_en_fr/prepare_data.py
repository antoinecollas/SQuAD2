import sys
sys.path.append("../../fastai/")
sys.path.append("../")
from fastai.text import *
from constants import *
from multiprocessing import Pool
import os
from functools import partial
import time
import pickle

'''
    Prepare data (WMT en_fr) for translation from English to French
'''

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
    print("time dictionaries=",total)
    # toktoi = lambda p: [stoi[o] for o in p]
    # texts = np.array([toktoi(p) for p in tok])
    return tok, itos, stoi

def main():
    print("====ENGLISH PHRASES====")
    tok_en, itos_en, stoi_en = prepare_texts(RAW_EN, lang='en')
    t0 = time.time()
    texts_en = np.array(Pool(NCPUS).map(Toktoi(stoi_en), tok_en))
    t1 = time.time()
    total = t1-t0
    print("time tok to int=",total)

    print("====FRENCH PHRASES====")
    tok_fr, itos_fr, stoi_fr = prepare_texts(RAW_FR, lang='fr')
    t0 = time.time()
    texts_fr = np.array(Pool(NCPUS).map(Toktoi(stoi_fr), tok_fr))
    t1 = time.time()
    total = t1-t0
    print("time tok to int=",total)

    print("====PICKLE====")
    t0 = time.time()
    with open(PREPROCESSED_EN_TEXTS, 'wb') as f:
        pickle.dump(texts_en, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PREPROCESSED_EN_STOI, 'wb') as f:
        pickle.dump(dict(stoi_en.items()), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PREPROCESSED_EN_ITOS, 'wb') as f:
        pickle.dump(itos_en, f, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    total = t1-t0
    print("time english pickle=",total)

    t0 = time.time()
    with open(PREPROCESSED_FR_TEXTS, 'wb') as f:
        pickle.dump(texts_fr, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PREPROCESSED_FR_STOI, 'wb') as f:
        pickle.dump(dict(stoi_fr.items()), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PREPROCESSED_FR_ITOS, 'wb') as f:
        pickle.dump(itos_fr, f, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    total = t1-t0
    print("time french pickle=",total)

if __name__ == "__main__":
    main()