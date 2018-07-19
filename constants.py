import torch
import os

PADDING_IDX = 0
BOS_IDX = 2
EOS_IDX = 3
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
UNKNOW_WORD = '_unk_'
PADDING_WORD = '_pad_'

#constants for the preparation of WMT en_fr
UNKNOW_WORD_IDX = 1
MAX_VOCAB = 100000
MIN_FREQ = 0

#constants for training
ENGLISH_FILENAME = "training-giga-fren/giga-fren.release2.fixed_100_000.en"
FRENCH_FILENAME = "training-giga-fren/giga-fren.release2.fixed_100_000.fr"
NB_EPOCH = 20
BATCH_SIZE = 100
MAX_SEQ = 100

#hardware
NCPUS=os.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model
PRETRAIN = False
PATH_WEIGHTS = "./weights"

#hyperparameters
NB_LAYERS = 2
NB_HEADS = 4
D_MODEL = 128
NB_NEURONS = 256