import torch
import os
class Constants():
    #hardware
    NCPUS=os.cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TYPE = torch.float32

    #constants for the preparation of dataset
    #/!\ the idx must be int in [0,3] and words must be different from vocabulary of the texts
    PADDING_IDX = 0
    UNKNOW_WORD_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    UNKNOW_WORD = '_unk_'
    PADDING_WORD = '_pad_'

    #files
    TRAIN_SUFFIX = ".train"
    TEST_SUFFIX = ".test"
    SOURCE_SUFFIX = ".en"
    TARGET_SUFFIX = ".fr"

    #bpe files
    CODES_FILE = "codes_file"
    VOCAB_FILE = "vocab"
    BPE_FILE = "BPE_FILE"

    #model
    TRAIN = True
    PRETRAIN = not TRAIN
    TEST = PRETRAIN
    WEIGHTS_FILE = "weights"

