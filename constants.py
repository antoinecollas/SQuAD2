import torch

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
NB_EPOCH = 100
BATCH_SIZE = 50
MAX_SEQ = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparameters
NB_LAYERS = 2
NB_HEADS = 4
D_MODEL = 128
NB_NEURONS = 256