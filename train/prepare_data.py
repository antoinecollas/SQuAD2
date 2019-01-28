import sys, os, time, math, random, argparse, glob
import numpy as np
from constants import Constants

'''
    Prepare data (WMT en_fr) for translation from English to French using byte pair encoding.
    https://arxiv.org/pdf/1508.07909.pdf
'''

def get_paths(path_source, path_target, constants, hyperparams):
    folder = os.path.join(os.path.dirname(path_source), hyperparams.SAVE_DATA_FOLDER)
    if not os.path.exists(folder):
        os.mkdir(folder)
    print('Preprocessed files saved in:', folder)
    filename_source = os.path.basename(path_source)
    filename_target = os.path.basename(path_target)
    paths = {
        'folder_preprocessed': folder,
        'source': path_source,
        'target': path_target,
        'source_train': os.path.join(folder, filename_source+constants.TRAIN_SUFFIX),
        'source_test': os.path.join(folder, filename_source+constants.TEST_SUFFIX),
        'target_train': os.path.join(folder, filename_target+constants.TRAIN_SUFFIX),
        'target_test': os.path.join(folder, filename_target+constants.TEST_SUFFIX),
        'codes_source': os.path.join(folder, constants.CODES_FILE+constants.SOURCE_SUFFIX),
        'codes_target': os.path.join(folder, constants.CODES_FILE+constants.TARGET_SUFFIX),
        'bpe_source_train': os.path.join(folder, constants.BPE_FILE+constants.SOURCE_SUFFIX+constants.TRAIN_SUFFIX),
        'bpe_target_train': os.path.join(folder, constants.BPE_FILE+constants.TARGET_SUFFIX+constants.TRAIN_SUFFIX),
        'bpe_source_test': os.path.join(folder, constants.BPE_FILE+constants.SOURCE_SUFFIX+constants.TEST_SUFFIX),
        'bpe_target_test': os.path.join(folder, constants.BPE_FILE+constants.TARGET_SUFFIX+constants.TEST_SUFFIX)
    }
    return paths

def slip_train_test(path_source, path_target, path_source_train, path_source_test, path_target_train, path_target_test, max_nb_phrases, train_split=0.8):
    with open(path_source,"r") as f:
        phrases_src = np.array(f.readlines())
    with open(path_target,"r") as f:
        phrases_tgt = np.array(f.readlines())

    nb_phrases = phrases_src.shape[0]
    nb_train = min(math.floor(nb_phrases*hyperparams.TRAIN_SPLIT), math.floor(max_nb_phrases*hyperparams.TRAIN_SPLIT))
    print("Nb phrases in training set:", nb_train)
    nb_test = min(math.floor(nb_phrases*(1-hyperparams.TRAIN_SPLIT)), math.floor(max_nb_phrases*(1-hyperparams.TRAIN_SPLIT)))
    print("Nb phrases in test set:", nb_test)
    idx = np.arange(nb_phrases)
    random.shuffle(idx)
    idx_train = idx[:nb_train]
    idx_test = idx[nb_train:nb_train+nb_test]
    train_phrases_src = phrases_src[idx_train]
    train_phrases_tgt = phrases_tgt[idx_train]
    test_phrases_src = phrases_src[idx_test]
    test_phrases_tgt = phrases_tgt[idx_test]

    with open(path_source_train,"w") as f:
        for phrase in train_phrases_src:
            f.write("%s" % phrase)
    with open(path_source_test,"w") as f:
        for phrase in test_phrases_src:
            f.write("%s" % phrase)
    with open(path_target_train,"w") as f:
        for phrase in train_phrases_tgt:
            f.write("%s" % phrase)
    with open(path_target_test,"w") as f:
        for phrase in test_phrases_tgt:
            f.write("%s" % phrase)

def learn_bpe(path, path_codes_file, num_operations):
    t0 = time.time()
    command = "subword-nmt learn-bpe -s " + str(num_operations) + " < " + path +  " > " + path_codes_file
    os.system(command)
    t1 = time.time()
    total = t1-t0
    print("time to learn bpe=",total)

def apply_bpe(path, path_codes_file, path_bpe):
    t0 = time.time()
    command = "subword-nmt apply-bpe -c " + path_codes_file + " < " + path +  " > " + path_bpe
    os.system(command)
    t1 = time.time()
    total = t1-t0
    print("time to apply bpe=",total)

def main(path_source, path_target, constants, hyperparams):
    paths = get_paths(path_source, path_target, constants, hyperparams)

    print("========SPLIT TO TRAIN AND TEST FILES========")
    slip_train_test(paths['source'], paths['target'], paths['source_train'], paths['source_test'], \
        paths['target_train'], paths['target_test'], hyperparams.MAX_NB_PHRASES, hyperparams.TRAIN_SPLIT)

    print("========LEARN BPE========")
    learn_bpe(paths['source_train'], paths['codes_source'], hyperparams.NUMP_OPS_BPE)
    learn_bpe(paths['target_train'], paths['codes_target'], hyperparams.NUMP_OPS_BPE)

    print("========APPLY BPE========")
    apply_bpe(paths['source_train'], paths['codes_source'], paths['bpe_source_train'])
    apply_bpe(paths['target_train'], paths['codes_target'], paths['bpe_target_train'])
    apply_bpe(paths['source_test'], paths['codes_source'], paths['bpe_source_test'])
    apply_bpe(paths['target_test'], paths['codes_target'], paths['bpe_target_test'])

    for fl in glob.glob(paths['folder_preprocessed']+constants.CODES_FILE+'*'):
        os.remove(fl)

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument("-s", dest="path_source", required=True,
                    help="path to the file contaning all the source data (train+test)",
                    type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-t", dest="path_target", required=True,
                    help="path to the file contaning all the target data (train+test)",
                    type=lambda x: is_valid_file(parser, x))
    parser.add_argument('--dev', dest="dev_mode", action='store_true',
                    help="flag used to debug")

    args = parser.parse_args()
    print('')
    if args.dev_mode:
        print('################DEV MODE################')
        from hyperparams_dev import Hyperparams
    else:
        from hyperparams import Hyperparams
    hyperparams = Hyperparams()
    constants = Constants()
    print('Source:', args.path_source)
    print('Target:', args.path_target)
    print('')

    main(args.path_source, args.path_target, constants, hyperparams)