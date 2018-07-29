import torch
import os
#hardware
NCPUS=os.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#constants for the preparation of WMT en_fr
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
NUMP_OPS_BPE = 10

#files
FOLDER = "training-giga-fren/"
TRAIN_SUFFIX = ".train"
TEST_SUFFIX = ".test"
EN_SUFFIX = ".en"
FR_SUFFIX = ".fr"
SIZE = "100" #number of pair of phrases
RAW = FOLDER + "giga-fren.release2.fixed_" + SIZE
#bpe files
CODES_FILE = FOLDER + "codes_file_" + SIZE + "_" + str(NUMP_OPS_BPE)
VOCAB_FILE = FOLDER + "vocab_" + SIZE + "_" + str(NUMP_OPS_BPE)
BPE = "BPE"
PREPROCESSED_TEXTS = FOLDER + "texts_" + SIZE + ".pickle"
PREPROCESSED_STOI = FOLDER + "stoi_" + SIZE + ".pickle"
PREPROCESSED_ITOS = FOLDER + "itos_" + SIZE + ".pickle"


#model
PRETRAIN = False
TRAIN = True
PATH_WEIGHTS = "./weights_" + SIZE
#hyperparameters
NB_LAYERS = 2
NB_HEADS = 2
D_MODEL = 64
NB_NEURONS = 128
WARMUP_STEPS = 100

#constants for training
TRAIN_SPLIT = 0.8
NB_EPOCH = 300
BATCH_SIZE = 10
MAX_SEQ = 2000

PREDICT_BATCH_SIZE = 50
