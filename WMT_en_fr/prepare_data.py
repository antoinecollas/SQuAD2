import sys, os, pickle, time, math, random, nltk, collections
sys.path.append("../")
import numpy as np
from constants import *
from multiprocessing import Pool
from functools import partial

'''
    Prepare data (WMT en_fr) for translation from English to French using BPE
    https://arxiv.org/pdf/1508.07909.pdf
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

def slip_train_test(filename_source, filename_target, train_suffix, test_suffix, train_split=0.8):
    with open(filename_source,"r") as f:
        phrases_src = np.array(f.readlines())
    with open(filename_target,"r") as f:
        phrases_tgt = np.array(f.readlines())
    nb_phrases = phrases_src.shape[0]
    nb_train = math.floor(nb_phrases*TRAIN_SPLIT)
    idx = list(range(nb_phrases))
    random.shuffle(idx)
    idx_train = idx[:nb_train]
    idx_test = idx[nb_train:]
    train_phrases_src = phrases_src[idx_train]
    train_phrases_tgt = phrases_tgt[idx_train]
    test_phrases_src = phrases_src[idx_test]
    test_phrases_tgt = phrases_tgt[idx_test]

    with open(filename_source+train_suffix,"w") as f:
        for phrase in train_phrases_src:
            f.write("%s" % phrase)
    with open(filename_target+train_suffix,"w") as f:
        for phrase in train_phrases_tgt:
            f.write("%s" % phrase)
    with open(filename_source+test_suffix,"w") as f:
        for phrase in test_phrases_src:
            f.write("%s" % phrase)
    with open(filename_target+test_suffix,"w") as f:
        for phrase in test_phrases_tgt:
            f.write("%s" % phrase)

def learn_apply_bpe(filename_source, filename_target, num_operations, codes_file, vocab_file_source, \
    vocab_file_target, vocabulary_threshold, file_bpe_source, file_bpe_target):
    t0 = time.time()
    command = "subword-nmt learn-joint-bpe-and-vocab --input " + filename_source +  " " \
        + filename_target + " -s " + str(num_operations) + " -o " + codes_file + \
        " --write-vocabulary " + vocab_file_source + " " + vocab_file_target
    os.system(command)
    t1 = time.time()
    total = t1-t0
    print("time to learn bpe=",total)
    t0 = time.time()
    command = "subword-nmt apply-bpe -c " + codes_file + " --vocabulary " + vocab_file_source + \
        " --vocabulary-threshold " + str(vocabulary_threshold) + " < " + filename_source + " > " + \
        file_bpe_source
    os.system(command)
    command = "subword-nmt apply-bpe -c " + codes_file + " --vocabulary " + vocab_file_target + \
        " --vocabulary-threshold " + str(vocabulary_threshold) + " < " + filename_target + " > " + \
        file_bpe_target
    os.system(command)
    t1 = time.time()
    total = t1-t0
    print("time to apply bpe=",total)

def get_itos_stoi(filename_bpe_source, filename_bpe_target):
    t0 = time.time()
    with open(filename_bpe_source,"r") as f:
        phrases_source = f.readlines()
    with open(filename_bpe_target,"r") as f:
        phrases_target = f.readlines()
    for phrase in phrases_source:
        phrase = phrase.split(" ")
    for phrase in phrases_target:
        phrase = phrase.split(" ")
    phrases = phrases_source + phrases_target
    t1 = time.time()
    total = t1-t0
    print("time split=",total)
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
    return phrases_source, phrases_target, itos, stoi

def main():
    print("========SPLIT TO TRAIN AND TEST FILES========")
    slip_train_test(RAW+EN_SUFFIX, RAW+FR_SUFFIX, TRAIN_SUFFIX, TEST_SUFFIX, TRAIN_SPLIT)

    print("========SUBWORDS========")
    learn_apply_bpe(RAW+EN_SUFFIX+TRAIN_SUFFIX, RAW+FR_SUFFIX+TRAIN_SUFFIX, NUMP_OPS_BPE, \
        CODES_FILE, VOCAB_FILE+EN_SUFFIX, VOCAB_FILE+FR_SUFFIX, MIN_FREQ, \
        BPE+EN_SUFFIX+TRAIN_SUFFIX, BPE+FR_SUFFIX+TRAIN_SUFFIX)

    print("====STOI AND ITOS====")
    tok_en, tok_fr, itos, stoi = get_itos_stoi(BPE+EN_SUFFIX+TRAIN_SUFFIX, BPE+FR_SUFFIX+TRAIN_SUFFIX)

    print("====TOK TO INT ENGLISH====")
    t0 = time.time()
    texts_en = np.array(Pool(NCPUS).map(Toktoi(stoi), tok_en))
    t1 = time.time()
    total = t1-t0
    print("time tok to int=",total)

    print("====TOK TO INT FRENCH====")
    t0 = time.time()
    texts_fr = np.array(Pool(NCPUS).map(Toktoi(stoi), tok_fr))
    t1 = time.time()
    total = t1-t0
    print("time tok to int=",total)

    print("====PICKLE====")
    t0 = time.time()
    with open(PREPROCESSED_TEXTS+EN_SUFFIX+TRAIN_SUFFIX, 'wb') as f:
        pickle.dump(texts_en, f, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    total = t1-t0
    print("time english pickle=",total)

    t0 = time.time()
    with open(PREPROCESSED_TEXTS+FR_SUFFIX+TRAIN_SUFFIX, 'wb') as f:
        pickle.dump(texts_fr, f, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    total = t1-t0
    print("time french pickle=",total)

    with open(PREPROCESSED_STOI, 'wb') as f:
        pickle.dump(dict(stoi.items()), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PREPROCESSED_ITOS, 'wb') as f:
        pickle.dump(itos, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()