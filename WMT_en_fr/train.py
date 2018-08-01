import sys, pickle, time, math, random, nltk
sys.path.append("../")
import numpy as np
from translator import *
from constants import *
from prepare_data import *
from sampler import *

def main():
    print("====LOAD PREPROCESSED DATA====")
    with open(PREPROCESSED_TEXTS+EN_SUFFIX+TRAIN_SUFFIX, 'rb') as f:
        texts_en = pickle.load(f)
    with open(PREPROCESSED_TEXTS+EN_SUFFIX+TEST_SUFFIX, 'rb') as f:
        test_texts_en = pickle.load(f)
    with open(PREPROCESSED_TEXTS+FR_SUFFIX+TRAIN_SUFFIX, 'rb') as f:
        texts_fr = pickle.load(f)
    with open(PREPROCESSED_TEXTS+FR_SUFFIX+TEST_SUFFIX, 'rb') as f:
        test_texts_fr = pickle.load(f)
    with open(PREPROCESSED_STOI, 'rb') as f:
        dicti = pickle.load(f)
        stoi = collections.defaultdict(unknow_word, dicti)
    with open(PREPROCESSED_ITOS, 'rb') as f:
        itos = pickle.load(f)

    print("========REMOVE LONGEST PHRASES========")
    #we remove french texts which are longer than MAX_SEQ (for memory and computation)
    sorted_idx = list(SortSampler(texts_fr, key=lambda x: len(texts_fr[x])))
    # print(itotok(itos, texts_en[sorted_idx[0]]))
    # print(itotok(itos, texts_fr[sorted_idx[1000]]))
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

    def pad_batch(batch, length=None):
        # t0 = time.time()
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
        # t1 = time.time()
        # print("pad batch time=", t1-t0)
        return result
    batches_idx = list(SortishSampler(texts_en, key=lambda x: len(texts_en[x]), bs=BATCH_SIZE))
    nb_texts = len(texts_en)
    nb_batches = nb_texts//BATCH_SIZE
    if nb_texts % BATCH_SIZE != 0:
        nb_batches+=1
    if nb_batches<2:
        raise ValueError('There must be at least 2 batches.')

    tr = Translator(vocabulary_size_in=len(stoi),vocabulary_size_out=len(stoi), share_weights=SHARE_WEIGHTS, max_seq=MAX_SEQ,nb_layers=NB_LAYERS,nb_heads=NB_HEADS,d_model=D_MODEL,nb_neurons=NB_NEURONS,warmup_steps=WARMUP_STEPS)
    if PRETRAIN:
        tr.load_state_dict(torch.load(PATH_WEIGHTS))
    tr.to(DEVICE, TYPE)
    print("Nb parameters=",tr.count_parameters())
    if TRAIN:
        tr.train(True)
        print("=======TRAINING=======")
        nb_train_steps = NB_EPOCH*nb_batches
        print("Nb epochs=",NB_EPOCH)
        print("Nb batches=",nb_batches)
        print("Nb train steps=",nb_train_steps)
        for k in range(NB_EPOCH):
            print("=======Epoch:=======",k)
            loss=0
            for l in range(nb_batches):
                # if l%(nb_batches//10)==0:
                # print("Batch:",l)
                X_batch = torch.from_numpy(pad_batch(texts_en[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
                Y_batch = torch.from_numpy(pad_batch(texts_fr[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
                # t0 = time.time()
                loss = loss + tr.fit(X_batch, Y_batch)
                # t1 = time.time()
                # print("time fit=", t1-t0)
            if k%50==0:
                torch.save(tr.state_dict(), PATH_WEIGHTS)
            print(loss/nb_batches)

    if TEST:
        print("=======EVALUATION=======")
        print("=======BLEU ON TRAIN SET=======")
        batches_idx = list(SortishSampler(texts_en, key=lambda x: len(texts_en[x]), bs=PREDICT_BATCH_SIZE))
        train_references = []
        train_hypotheses = []
        nb_texts = len(texts_en)
        nb_batches = nb_texts//PREDICT_BATCH_SIZE
        itotok_fr = Itotok(itos)
        for l in range(nb_batches):
            print("Batch:",l)
            X_batch = torch.from_numpy(pad_batch(texts_en[batches_idx[l*PREDICT_BATCH_SIZE:(l+1)*PREDICT_BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
            Y_batch = texts_fr[batches_idx[l*PREDICT_BATCH_SIZE:(l+1)*PREDICT_BATCH_SIZE]]
            for i in range(Y_batch.shape[0]):
                train_references.append([itotok_fr(list(Y_batch[i]))])
            hypotheses = tr.predict(X_batch)
            for i in range(len(hypotheses)):
                train_hypotheses.append(itotok_fr(hypotheses[i]))

        def subwords_to_string(subwords):
            string = ""
            for subword in subwords:
                if subword[-2:] == "@@":
                    string += subword[:-2]
                elif subword != PADDING_WORD:
                    string += subword + " "
            return string

        for i, phrases in enumerate(zip(train_references, train_hypotheses)):
            train_references[i][0] = subwords_to_string(phrases[0][0])
            train_hypotheses[i] = subwords_to_string(phrases[1])

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
        itotok_fr = Itotok(itos)
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
            test_references[i][0] = subwords_to_string(phrases[0][0])
            test_hypotheses[i] = subwords_to_string(phrases[1])

        for i, phrases in enumerate(zip(test_references, test_hypotheses)):
            print(phrases[0])
            print(phrases[1])
            print("")
            if i==5:
                break
        BLEU = nltk.translate.bleu_score.corpus_bleu(test_references, test_hypotheses)
        print(BLEU)

if __name__ == "__main__":
    main()