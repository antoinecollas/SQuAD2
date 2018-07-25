import sys
sys.path.append("../../fastai/")
sys.path.append("../")
from fastai.text import *
import pickle
import time
import math
import random
from translator import *
from constants import *
from prepare_data import *
import nltk

print("====LOAD PREPROCESSED DATA====")
t0 = time.time()
with open(PREPROCESSED_EN_TEXTS, 'rb') as f:
    texts_en = pickle.load(f)
with open(PREPROCESSED_EN_STOI, 'rb') as f:
    dict_en = pickle.load(f)
    stoi_en = collections.defaultdict(unknow_word, dict_en)
with open(PREPROCESSED_EN_ITOS, 'rb') as f:
    itos_en = pickle.load(f)
t1 = time.time()
total = t1-t0
print("time to load english=",total)

t0 = time.time()
with open(PREPROCESSED_FR_TEXTS, 'rb') as f:
    texts_fr = pickle.load(f)
with open(PREPROCESSED_FR_STOI, 'rb') as f:
    dict_fr = pickle.load(f)
    stoi_fr = collections.defaultdict(unknow_word, dict_fr)
with open(PREPROCESSED_FR_ITOS, 'rb') as f:
    itos_fr = pickle.load(f)
t1 = time.time()
total = t1-t0
print("time to load french=",total)

print("========REMOVE LONGEST PHRASES========")
#we remove french texts which are longer than MAX_SEQ (for memory and computation)
sorted_idx = list(SortSampler(texts_fr, key=lambda x: len(texts_fr[x])))
# print(itotok(itos_en, texts_en[sorted_idx[0]]))
# print(itotok(itos_fr, texts_fr[sorted_idx[1000]]))
to_remove = []
for idx in sorted_idx:
    length = len(texts_fr[idx])
    if length <= MAX_SEQ:
        break
    else:
        to_remove.append(idx)
to_remove = np.array(to_remove)
print(len(to_remove), "phases removed")
texts_en = np.delete(texts_en, to_remove)
texts_fr = np.delete(texts_fr, to_remove)

print("========PREPARATION OF TRAIN AND TEST DATASETS========")
nb_texts = len(texts_en)
nb_train = math.floor(nb_texts*TRAIN_SPLIT)
idx = list(range(nb_texts))
random.shuffle(idx)
idx_train = idx[:nb_train]
idx_test = idx[nb_train:]
train_texts_en = texts_en[idx_train]
# print(train_texts_en.shape)
train_texts_fr = texts_fr[idx_train]
# print(train_texts_en[:2])
test_texts_en = texts_en[idx_test]
# print(test_texts_en.shape)
# print(test_texts_en[:2])
test_texts_fr = texts_fr[idx_test]

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

batches_idx = list(SortishSampler(train_texts_en, key=lambda x: len(train_texts_en[x]), bs=BATCH_SIZE))
nb_texts = len(train_texts_en)
nb_batches = nb_texts//BATCH_SIZE
if nb_batches<=2:
    raise ValueError('There must be at least 2 batches.')

tr = Translator(vocabulary_size_in=len(stoi_en),vocabulary_size_out=len(stoi_fr),max_seq=MAX_SEQ,nb_layers=NB_LAYERS,nb_heads=NB_HEADS,d_model=D_MODEL,nb_neurons=NB_NEURONS)
if PRETRAIN:
    tr.load_state_dict(torch.load(PATH_WEIGHTS))
tr.to(DEVICE)
print("Nb parameters=",tr.count_parameters())
if TRAIN:
    tr.train(True)
    print("=======TRAINING=======")
    nb_train_steps = NB_EPOCH*nb_batches
    loss=0
    print("Nb train steps=",nb_train_steps)
    for k in range(NB_EPOCH):
        print("=======Epoch:=======",k)
        for l in range(nb_batches):
            if l%100==0:
                print("Batch:",l)
                print(loss/100)
                loss=0
            X_batch = torch.from_numpy(pad_batch(train_texts_en[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
            Y_batch = torch.from_numpy(pad_batch(train_texts_fr[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
            loss = loss + tr.fit(X_batch, Y_batch)
        torch.save(tr.state_dict(), PATH_WEIGHTS)

print("=======EVALUATION=======")
print("=======BLEU ON TRAIN SET=======")
batches_idx = list(SortishSampler(train_texts_en, key=lambda x: len(train_texts_en[x]), bs=PREDICT_BATCH_SIZE))
train_references = []
train_hypotheses = []
nb_texts = len(train_texts_en)
nb_batches = nb_texts//PREDICT_BATCH_SIZE
itotok_fr = Itotok(itos_fr)
for l in range(nb_batches):
    print("Batch:",l)
    X_batch = torch.from_numpy(pad_batch(train_texts_en[batches_idx[l*PREDICT_BATCH_SIZE:(l+1)*PREDICT_BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
    Y_batch = train_texts_fr[batches_idx[l*PREDICT_BATCH_SIZE:(l+1)*PREDICT_BATCH_SIZE]]
    for i in range(Y_batch.shape[0]):
        train_references.append([itotok_fr(list(Y_batch[i]))])
    hypotheses = tr.predict(X_batch)
    for i in range(len(hypotheses)):
        train_hypotheses.append(itotok_fr(hypotheses[i]))
for i, phrases in enumerate(zip(train_references, train_hypotheses)):
    print(phrases[0])
    print(phrases[1])
    print("")
    if i==5:
        break
BLEU = nltk.translate.bleu_score.corpus_bleu(train_references, train_hypotheses)
print(BLEU)

print("=======BLEU ON TEST SET=======")
batches_idx = list(SortishSampler(test_texts_en, key=lambda x: len(test_texts_en[x]), bs=PREDICT_BATCH_SIZE))
test_references = []
test_hypotheses = []
nb_texts = len(test_texts_en)
nb_batches = nb_texts//PREDICT_BATCH_SIZE
itotok_fr = Itotok(itos_fr)
for l in range(nb_batches):
    print("Batch:",l)
    X_batch = torch.from_numpy(pad_batch(test_texts_en[batches_idx[l*PREDICT_BATCH_SIZE:(l+1)*PREDICT_BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
    Y_batch = test_texts_fr[batches_idx[l*PREDICT_BATCH_SIZE:(l+1)*PREDICT_BATCH_SIZE]]
    for i in range(Y_batch.shape[0]):
        test_references.append([itotok_fr(list(Y_batch[i]))])
    hypotheses = tr.predict(X_batch)
    for i in range(len(hypotheses)):
        test_hypotheses.append(itotok_fr(hypotheses[i]))
for i, phrases in enumerate(zip(test_references, test_hypotheses)):
    print(phrases[0])
    print(phrases[1])
    print("")
    if i==5:
        break
BLEU = nltk.translate.bleu_score.corpus_bleu(test_references, test_hypotheses)
print(BLEU)