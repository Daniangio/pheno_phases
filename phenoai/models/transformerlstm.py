import logging
import os
from phenoai.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy
import numpy as np

logger = logging.getLogger()

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Linear(vocab_size, d_model) # nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 366):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x, device):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).to(device)
        return x

    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        # output = self.out(concat)
    
        return concat
    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        #[batch_size, 1, 366, 366]
        if mask is not None:
            mask = mask.unsqueeze(1)
            #print('ATTENTION MASK', mask)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

    
# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.ff(x2))
        return x2
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.01):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()
        
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        # x = x + self.dropout_3(self.ff(x2))
        return x2
    
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_seq_len):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask, device):
        # [1, 366, INPUT_SIZE]
        x = self.embed(src)
        # [1, 366, d_model]
        x = self.pe(x, device)
        # [1, 366, d_model]
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_seq_len):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=max_seq_len)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    
    def forward(self, trg, e_outputs, src_mask, trg_mask, device):
        x = self.embed(trg)
        x = self.pe(x, device)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, num_layers=2, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size),
                            torch.zeros(num_layers, 1, self.hidden_layer_size))
    
    def reset_hidden_state(self, device):
        self.hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_layer_size).to(device),
                            torch.zeros(self.num_layers, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = torch.sigmoid(self.linear(lstm_out.view(len(input_seq), -1)))
        return predictions


class TransformerLSTM(BaseModel):
    def __init__(self, version, input_features, output_features, d_model=32, N=2, heads=1, max_seq_len=366, **kwargs):
        super().__init__(version, input_features, output_features)
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(len(input_features), d_model, N, heads, max_seq_len)
        self.decoder = LSTM(input_size=d_model, output_size=len(output_features))
        self.d_model = d_model
    
    def reset_hidden_state(self, device):
        self.decoder.reset_hidden_state(device)
        
    def forward(self, src, trg, src_mask=None, trg_mask=None, device='cuda'):
        e_outputs = self.encoder(src, src_mask, device)
        return self.decoder(e_outputs.view(-1, self.d_model))
    
    def get_weights_path(self, root, place, variety):
        return os.path.join(root, f'transformer_lstm_{place}_{variety}_{self.version}.pt')
    
    def run_inference(self, src, device):
        src_input = F.pad(src, (0, 0, 0, self.max_seq_len - src.size(1)), 'constant', -2)
        src_mask = self.create_masks(src_input.squeeze(0).transpose(0,1), device)
        self.reset_hidden_state(device)
        y_pred = self(src_input, None, src_mask=src_mask, trg_mask=None, device=device)
        y_pr = y_pred.squeeze(0).detach().cpu().numpy()
        return y_pr
    
    @staticmethod
    def create_masks(src, device, pad=-2): # Magheggione dell' and bit a bit
        #src_mask = (src != pad).unsqueeze(-2).to(device)
        #size = src.size(1) # get seq_len for matrix
        #np_mask = Variable(torch.from_numpy(np.ones((1, size, size)).astype('uint8')) == 1).to(device)
        #src_mask = src_mask & src_mask.transpose(1,2) & np_mask
        if src is not None:
            src_mask = (src != pad).unsqueeze(-2)
            size = src.size(1) # get seq_len for matrix
            np_mask = nopeak_mask(size).to(device)
            src_mask = src_mask & src_mask.transpose(1,2) & np_mask
            src_mask[0, :, 0] = True
        else:
            src_mask = None
        return src_mask[:1, ...]

def nopeak_mask(size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8') # k=-30 rende il modello "cieco" al valore dell'output degli ultimi 30 valori (giorni)
        np_mask =  Variable(torch.from_numpy(np_mask) == 0)
        return np_mask