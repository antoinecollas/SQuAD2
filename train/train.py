import sys, pickle, time, math, random, nltk, argparse, os
import numpy as np
from sampler import *
from data_loader import DataLoader
from transformer.translator import *

def get_paths(folder, constants, hyperparams):
    folder = os.path.join(folder, hyperparams.SAVE_DATA_FOLDER)
    if not os.path.exists(folder):
        raise ValueError('Folder doesn\'t exist')
    print('Folder:', folder)
    filename_source = 'baseline-1M.en'
    filename_target = 'baseline-1M.fr'
    paths = {
        'folder_preprocessed': folder,
        'bpe_source_train': os.path.join(folder, constants.BPE_FILE+constants.SOURCE_SUFFIX+constants.TRAIN_SUFFIX),
        'bpe_target_train': os.path.join(folder, constants.BPE_FILE+constants.TARGET_SUFFIX+constants.TRAIN_SUFFIX),
        'bpe_source_test': os.path.join(folder, constants.BPE_FILE+constants.SOURCE_SUFFIX+constants.TEST_SUFFIX),
        'bpe_target_test': os.path.join(folder, constants.BPE_FILE+constants.TARGET_SUFFIX+constants.TEST_SUFFIX)
    }
    return paths

def main(folder, constants, hyperparams):
    paths = get_paths(folder, constants, hyperparams)
    paths_training = {
        'bpe_source': paths['bpe_source_train'],
        'bpe_target': paths['bpe_target_train'],
    }
    data_training = DataLoader(paths_training, constants, hyperparams)
    print('Nb training phrases:', len(data_training))

    paths_eval = {
        'bpe_source': paths['bpe_source_test'],
        'bpe_target': paths['bpe_target_test'],
    }
    data_eval = DataLoader(paths_eval, constants, hyperparams)
    print('Nb eval phrases:', len(data_eval))

    tr = Translator(
        vocabulary_size_in=len(data_training.stoi),
        vocabulary_size_out=len(data_training.stoi),
        constants=constants,
        share_weights=hyperparams.SHARE_WEIGHTS,
        max_seq=hyperparams.MAX_SEQ,
        nb_layers=hyperparams.NB_LAYERS,
        nb_heads=hyperparams.NB_HEADS,
        d_model=hyperparams.D_MODEL,
        nb_neurons=hyperparams.NB_NEURONS,
        warmup_steps=hyperparams.WARMUP_STEPS)
    
    tr.to(constants.DEVICE, constants.TYPE)
    print("Nb parameters=",tr.count_parameters())
    
    if constants.TRAIN:
        tr.train(True)
        print("=======TRAINING=======")
        nb_train_steps = hyperparams.NB_EPOCH*data_training.nb_batches
        print("Nb epochs=", hyperparams.NB_EPOCH)
        print("Nb batches=", data_training.nb_batches)
        print("Nb train steps=", nb_train_steps)
        # tr.scheduler.plot_lr(nb_train_steps=nb_train_steps)
        tr.fit(hyperparams.NB_EPOCH, data_training, data_eval)
        tr.plot_training_loss()

    # if constants.PRETRAIN:
    #     tr.load_state_dict(torch.load(constants.WEIGHTS_FILE))

    # if constants.TEST:
    #     def evaluate(texts_src, texts_tgt, verbose=True):
    #         data_iter = DataLoader(texts_src, texts_tgt, constants.PREDICT_BATCH_SIZE, constants.DEVICE, pad_Y_batch=False)
    #         train_references = []
    #         train_hypotheses = []
    #         itotok_fr = Itotok(itos)
    #         for l, (X_batch, Y_batch) in enumerate(data_iter):
    #             print("Batch:",l)
    #             for i in range(Y_batch.shape[0]):
    #                 train_references.append([itotok_fr(list(Y_batch[i]))])
    #             hypotheses = tr.translate(X_batch)
    #             for i in range(len(hypotheses)):
    #                 train_hypotheses.append(itotok_fr(hypotheses[i]))

    #         def subwords_to_string(subwords):
    #             string = ""
    #             for subword in subwords:
    #                 if subword[-2:] == "@@":
    #                     string += subword[:-2]
    #                 elif subword != PADDING_WORD:
    #                     string += subword + " "
    #             return string

    #         for i, phrases in enumerate(zip(train_references, train_hypotheses)):
    #             train_references[i][0] = subwords_to_string(phrases[0][0])
    #             train_hypotheses[i] = subwords_to_string(phrases[1])

    #         if verbose:
    #             for i, phrases in enumerate(zip(train_references, train_hypotheses)):
    #                 print(phrases[0])
    #                 print(phrases[1])
    #                 print("")
    #                 if i==5:
    #                     break

    #         BLEU = nltk.translate.bleu_score.corpus_bleu(train_references, train_hypotheses)
    #         return BLEU
        
    #     print("=======EVALUATION=======")
    #     print("=======BLEU ON TRAIN SET=======")
    #     bleu_train = evaluate(texts_en, texts_fr)
    #     print(bleu_train)

    #     print("=======BLEU ON TEST SET=======")
    #     bleu_test = evaluate(test_texts_en, test_texts_fr)
    #     print(bleu_test)

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a machine translation.')
    parser.add_argument("-f", dest="folder_dataset", required=True,
                    help="path to the folder contaning all the data",
                    type=lambda x: is_valid_file(parser, x))
    parser.add_argument('--dev', dest="dev_mode", action='store_true',
                    help="flag used to debug")
    args = parser.parse_args()

    print('')
    from constants import Constants
    if args.dev_mode:
        print('################DEV MODE################')
        from hyperparams_dev import Hyperparams
    else:
        from hyperparams import Hyperparams
    constants = Constants()
    hyperparams = Hyperparams()
    
    main(args.folder_dataset, constants, hyperparams)