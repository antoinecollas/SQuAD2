import sys, pickle, time, math, random, nltk
sys.path.append("../")
import numpy as np
from translator import *
from constants import *
from prepare_data import *
from sampler import *
from data_loader import DataLoader

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
    def longest(texts_fr):
        sorted_idx = list(SortSampler(texts_fr, key=lambda x: len(texts_fr[x])))
        to_remove = []
        for idx in sorted_idx:
            length = len(texts_fr[idx])
            if length <= MAX_SEQ:
                break
            else:
                to_remove.append(idx)
        to_remove = np.array(to_remove)
        return to_remove
    to_remove = longest(texts_fr)
    print(len(to_remove), "phases removed in training set")
    texts_en = np.delete(texts_en, to_remove)
    texts_fr = np.delete(texts_fr, to_remove)
    to_remove = longest(test_texts_fr)
    print(len(to_remove), "phases removed in test set")
    test_texts_en = np.delete(test_texts_en, to_remove)
    test_texts_fr = np.delete(test_texts_fr, to_remove)

    tr = Translator(vocabulary_size_in=len(stoi),vocabulary_size_out=len(stoi), share_weights=SHARE_WEIGHTS, max_seq=MAX_SEQ,nb_layers=NB_LAYERS,nb_heads=NB_HEADS,d_model=D_MODEL,nb_neurons=NB_NEURONS,warmup_steps=WARMUP_STEPS)
    
    if PRETRAIN:
        tr.load_state_dict(torch.load(PATH_WEIGHTS))

    tr.to(DEVICE, TYPE)
    print("Nb parameters=",tr.count_parameters())
    
    if TRAIN:
        tr.train(True)
        data_iter = DataLoader(texts_en, texts_fr, BATCH_SIZE, DEVICE)
        print("=======TRAINING=======")
        nb_train_steps = NB_EPOCH*data_iter.nb_batches
        print("Nb epochs=",NB_EPOCH)
        print("Nb batches=",data_iter.nb_batches)
        print("Nb train steps=",nb_train_steps)
        # tr.scheduler.plot_lr(nb_train_steps=nb_train_steps)
        tr.fit(data_iter, nb_epoch=NB_EPOCH)
        tr.plot_training_loss()

    if TEST:
        def evaluate(texts_src, texts_tgt, verbose=True):
            data_iter = DataLoader(texts_src, texts_tgt, PREDICT_BATCH_SIZE, DEVICE, pad_Y_batch=False)
            train_references = []
            train_hypotheses = []
            itotok_fr = Itotok(itos)
            for l, (X_batch, Y_batch) in enumerate(data_iter):
                print("Batch:",l)
                for i in range(Y_batch.shape[0]):
                    train_references.append([itotok_fr(list(Y_batch[i]))])
                hypotheses = tr.translate(X_batch)
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

            if verbose:
                for i, phrases in enumerate(zip(train_references, train_hypotheses)):
                    print(phrases[0])
                    print(phrases[1])
                    print("")
                    if i==5:
                        break

            BLEU = nltk.translate.bleu_score.corpus_bleu(train_references, train_hypotheses)
            return BLEU
        
        print("=======EVALUATION=======")
        print("=======BLEU ON TRAIN SET=======")
        bleu_train = evaluate(texts_en, texts_fr)
        print(bleu_train)

        print("=======BLEU ON TEST SET=======")
        bleu_test = evaluate(test_texts_en, test_texts_fr)
        print(bleu_test)

if __name__ == "__main__":
    main()