import torch
import os
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
MAX_VOCAB = 1000000
MIN_FREQ = 0
NUMP_OPS_BPE = 100

#files
FOLDER = "europarl/"
TRAIN_SUFFIX = ".train"
TEST_SUFFIX = ".test"
EN_SUFFIX = ".en"
FR_SUFFIX = ".fr"
SIZE = "100" #number of pair of phrases
RAW = FOLDER + "europarl_" + SIZE
#bpe files
CODES_FILE = FOLDER + "codes_file_" + SIZE + "_" + str(NUMP_OPS_BPE)
VOCAB_FILE = FOLDER + "vocab_" + SIZE + "_" + str(NUMP_OPS_BPE)
BPE = FOLDER + "BPE"
PREPROCESSED_TEXTS = FOLDER + "texts_" + SIZE + ".pickle"
PREPROCESSED_STOI = FOLDER + "stoi_" + SIZE + ".pickle"
PREPROCESSED_ITOS = FOLDER + "itos_" + SIZE + ".pickle"


#model
TRAIN = True
PRETRAIN = not TRAIN
TEST = PRETRAIN
PATH_WEIGHTS = "./weights_" + SIZE
#hyperparameters
SHARE_WEIGHTS = True
NB_LAYERS = 2
NB_HEADS = 8
D_MODEL = 64
NB_NEURONS = 128
WARMUP_STEPS = 100

#constants for training
TRAIN_SPLIT = 0.8
NB_EPOCH = 1000
BATCH_SIZE = 20
MAX_SEQ = 256

PREDICT_BATCH_SIZE = 10
