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

#files
FOLDER = "training-giga-fren/"
SIZE = "10_000" #number of pair of phrases
RAW_EN = FOLDER + "giga-fren.release2.fixed_" + SIZE + ".en"
RAW_FR = FOLDER + "giga-fren.release2.fixed_" + SIZE + ".fr"
PREPROCESSED_EN_TEXTS = FOLDER + "texts_" + SIZE + ".en.pickle"
PREPROCESSED_EN_STOI = FOLDER + "stoi_" + SIZE + ".en.pickle"
PREPROCESSED_EN_ITOS = FOLDER + "itos_" + SIZE + ".en.pickle"
PREPROCESSED_FR_TEXTS = FOLDER + "texts_" + SIZE + ".fr.pickle"
PREPROCESSED_FR_STOI = FOLDER + "stoi_" + SIZE + ".fr.pickle"
PREPROCESSED_FR_ITOS = FOLDER + "itos_" + SIZE + ".fr.pickle"

#model
PRETRAIN = False
TRAIN = True
PATH_WEIGHTS = "./weights_" + SIZE
#hyperparameters
NB_LAYERS = 1
NB_HEADS = 2
D_MODEL = 32
NB_NEURONS = 64
WARMUP_STEPS = 10

#constants for training
TRAIN_SPLIT = 0.8
NB_EPOCH = 300
BATCH_SIZE = 50
MAX_SEQ = D_MODEL

PREDICT_BATCH_SIZE = 50
