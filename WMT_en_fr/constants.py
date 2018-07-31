import torch
import os
#hardware
NCPUS=os.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE = torch.float32

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
NUMP_OPS_BPE = 500

#files
FOLDER = "training-giga-fren/"
TRAIN_SUFFIX = ".train"
TEST_SUFFIX = ".test"
EN_SUFFIX = ".en"
FR_SUFFIX = ".fr"
SIZE = "10_000" #number of pair of phrases
RAW = FOLDER + "giga-fren.release2.fixed_" + SIZE
#bpe files
CODES_FILE = FOLDER + "codes_file_" + SIZE + "_" + str(NUMP_OPS_BPE)
VOCAB_FILE = FOLDER + "vocab_" + SIZE + "_" + str(NUMP_OPS_BPE)
BPE = "BPE"
PREPROCESSED_TEXTS = FOLDER + "texts_" + SIZE + ".pickle"
PREPROCESSED_STOI = FOLDER + "stoi_" + SIZE + ".pickle"
PREPROCESSED_ITOS = FOLDER + "itos_" + SIZE + ".pickle"


#model
TRAIN = True
PRETRAIN = not TRAIN
TEST = False
PATH_WEIGHTS = "./weights_" + SIZE
#hyperparameters
SHARE_WEIGHTS = True
NB_LAYERS = 4
NB_HEADS = 2
D_MODEL = 128
NB_NEURONS = 256
WARMUP_STEPS = 2000

#constants for training
TRAIN_SPLIT = 0.8
NB_EPOCH = 100000
BATCH_SIZE = 350
MAX_SEQ = 150

PREDICT_BATCH_SIZE = 200
