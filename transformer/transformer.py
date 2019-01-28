import torch, time
import torch.nn as nn
import torch.nn.functional as F
from transformer.layers import Embedding, get_mask
from transformer.encoder import Encoder
from transformer.decoder import Decoder
# from constants import *

# torch.manual_seed(1)

class Transformer(nn.Module):
    def __init__(self, vocabulary_size_in, vocabulary_size_out, constants, share_weights=True, max_seq=100, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.constants = constants
        self.max_seq = max_seq
        self.EmbeddingSrc = Embedding(vocabulary_size=vocabulary_size_in, d_model=d_model, constants=constants)
        self.EmbeddingTgt = Embedding(vocabulary_size=vocabulary_size_out, d_model=d_model, constants=constants)
        self.Encoder = Encoder(nb_layers=nb_layers, nb_heads=nb_heads, d_model=d_model, nb_neurons=nb_neurons, dropout=dropout)
        self.Decoder = Decoder(nb_layers=nb_layers, nb_heads=nb_heads, d_model=d_model, nb_neurons=nb_neurons, dropout=dropout)
        self.Linear = nn.Linear(d_model, vocabulary_size_out, bias=False)
        if share_weights:
            self.EmbeddingSrc.lookup_table.weight = self.Linear.weight
            self.EmbeddingTgt.lookup_table.weight = self.Linear.weight

    def forward_encoder(self, src):
        '''
        Arg:
            src: tensor(nb_texts, nb_tokens)
        Output:
            tensor(nb_texts, nb_tokens, d_model)
        '''
        mask = get_mask(src,src, self.constants.PADDING_IDX, self.constants.DEVICE)
        enc = self.EmbeddingSrc(src)
        enc = self.Encoder(enc,mask)
        return enc

    def forward_decoder(self, src, enc, target):
        '''
        Arg:
            src: tensor(nb_texts, nb_tokens)
            enc: tensor(nb_texts, nb_tokens, d_model)
            target: tensor(nb_texts, nb_tokens)
        Output:
            tensor(nb_texts, nb_tokens, vocabulary_size_out)
        '''
        mask1 = get_mask(target,target, self.constants.PADDING_IDX, self.constants.DEVICE,avoid_subsequent_info=True)
        mask2 = get_mask(target,src, self.constants.PADDING_IDX, self.constants.DEVICE)
        output = self.EmbeddingTgt(target)
        output = self.Decoder(enc, output, mask1, mask2)
        output = self.Linear(output)
        # output = F.softmax(output, dim=-1)
        return output

    def forward(self, src, target):
        '''
        Arg:
            src: tensor(nb_texts, nb_tokens)
            target: tensor(nb_texts, nb_tokens)
        Output:
            tensor(nb_texts, nb_tokens, vocabulary_size_out)
        '''
        enc = self.forward_encoder(src)
        output = self.forward_decoder(src, enc, target)
        return output