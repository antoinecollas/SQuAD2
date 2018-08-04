import torch
import torch.nn as nn
from layers import MultiHeadAttention, PositionWiseFeedForward, get_mask

# torch.manual_seed(1)

class Encoder(nn.Module):
    def __init__(self, nb_layers=6, nb_heads=8, d_model=512, nb_neurons = 2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.nb_layers = nb_layers
        self.MultiHeadAttention = nn.ModuleList([MultiHeadAttention(nb_heads, d_model, dropout) for _ in range(self.nb_layers)])
        self.PositionWiseFeedForward = nn.ModuleList([PositionWiseFeedForward(d_model, nb_neurons, dropout) for _ in range(self.nb_layers)])

    def forward(self, X, mask=None):
        '''
        Arg:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        Output:
            tensor(nb_texts, nb_tokens, d_model(=size of one token))
        '''
        output = X
        for i in range(self.nb_layers):
            output = self.MultiHeadAttention[i].forward(output,output,output,mask)
            output = self.PositionWiseFeedForward[i].forward(output)
        return output

