import sys, pickle, time, math, random, nltk, argparse, os
import numpy as np
from sampler import *
from data_loader import *
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

    bpe_src = read_file(paths['bpe_source_train'], tokenize=True)
    bpe_tgt = read_file(paths['bpe_target_train'], tokenize=True)
    itos = get_itos(bpe_src, bpe_tgt, constants, hyperparams)
    stoi = get_stoi_from_itos(itos, constants)
    data_training = DataLoader(bpe_src, bpe_tgt, constants, hyperparams, itos, stoi)
    print('Nb training phrases:', len(data_training))

    bpe_src = read_file(paths['bpe_source_test'], tokenize=True)
    bpe_tgt = read_file(paths['bpe_target_test'], tokenize=True)
    data_eval = DataLoader(bpe_src, bpe_tgt, constants, hyperparams, itos, stoi)
    print('Nb eval phrases:', len(data_eval))

    tr = Translator(
        vocabulary_size_in=len(stoi),
        vocabulary_size_out=len(stoi),
        constants=constants,
        hyperparams=hyperparams)
    
    tr.to(constants.DEVICE)
    print("Nb parameters=",tr.count_parameters())
    print("=======TRAINING=======")
    tr.train(True)
    nb_train_steps = hyperparams.NB_EPOCH*data_training.nb_batches
    print("Nb epochs=", hyperparams.NB_EPOCH)
    print("Nb batches=", data_training.nb_batches)
    print("Nb train steps=", nb_train_steps)
    tr.fit(hyperparams.NB_EPOCH,
        data_training,
        data_eval
    )

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