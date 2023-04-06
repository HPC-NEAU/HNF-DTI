import math
import torch
from torch import nn
from torch_multi_head_attention import MultiHeadAttention
from torch.autograd import Variable


class Add_Norm(nn.Module):
    """After the residual connection, the layer is normalized"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(Add_Norm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y) )

class PositionWiseFFN(nn.Module):
    #Location-based feedforward networks
    #input_Dim = output_Dim
    def __init__(self, input_Dim, hidden,**kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(input_Dim, hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden, input_Dim)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('x.size(1):',x.size(1))
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    """transformer encoder block  q_k_v_size-> embedSize """
    def __init__(self, embed_Size, ffn_hidden,
        norm_shape, heads,  
                 #ffn_num_input
        dropout, use_bias=True, **kwargs):

        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embed_Size, heads, use_bias)
        self.addnorm1 = Add_Norm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(embed_Size, ffn_hidden)
        self.addnorm2 = Add_Norm(norm_shape, dropout)
    def forward(self, X, mask=None):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        Y = self.addnorm1(X, self.attention(X, X, X, mask))
        return self.addnorm2(Y, self.ffn(Y))

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_Size, ffn_hiddens, max_len, heads, norm_shape, n_layers, dropout, use_bias=True, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embed_Size = embed_Size
        self.embedding = nn.Embedding(vocab_size, embed_Size)
        self.pos_encoding = PositionalEncoding(embed_Size, dropout, max_len=max_len)
        self.blks = nn.Sequential()
        for i in range(n_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(embed_Size, ffn_hiddens,
                                              norm_shape, heads, dropout))
    def forward(self, X, mask=None, *args):



        for layer in self.blks:
            X = layer(X, mask)

        return X
    
  