import math
import torch
from torch import nn
from torch_multi_head_attention import MultiHeadAttention
from torch.autograd import Variable

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
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
class PositionWiseFFN(nn.Module):
    
    #input_Dim = output_Dim
    def __init__(self, input_Dim, hidden,**kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(input_Dim, hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(input_Dim, input_Dim)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))



class Add_Norm(nn.Module):
    """After the residual connection,the layer is normalized"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(Add_Norm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y) )

class EncoderBlock(nn.Module):
    """transformer encoder block  q_k_v_size-> embedSize """
    def __init__(self, embed_Size, ffn_hidden,
        norm_shape, heads,  #ffn_num_input
        dropout, use_bias=True, **kwargs):

        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embed_Size, heads, use_bias)
        self.addnorm1 = Add_Norm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(embed_Size, ffn_hidden)
        self.addnorm2 = Add_Norm(norm_shape, dropout)
    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
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
    def forward(self, X, *args):
        
        for layer in self.blks:
            X = layer(X)
        return X

class DecoderBlock(nn.Module):
    def __init__(self, embed_Size, ffn_hidden, norm_shape,heads,  droupout, use_mask=None):
        super(DecoderBlock, self).__init__()
        self.attention1 = MultiHeadAttention(embed_Size, heads, droupout)
        self.addNorm1 = Add_Norm(norm_shape, droupout)
        self.attention2 = MultiHeadAttention(embed_Size, heads, droupout)
        self.addNorm2 = Add_Norm(norm_shape, droupout)
        self.ffn = PositionWiseFFN(embed_Size, ffn_hidden)
        self.addNorm3 = Add_Norm(norm_shape, droupout)
        self.mask = use_mask
    def forward(self, x, x2):
        
        X1 = self.attention1(x, x, x)
        Y1 = self.addNorm1(x, X1)
        
        Y2= self.attention2(x2, x2, Y1, mask=self.mask)
        Z = self.addNorm2(Y1, Y2)

        return self.addNorm3(Z, self.ffn(Z))


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_Size, ffn_hidden, max_len, heads, norm_shape,  n_layers, dropout, mask):
        super(Decoder, self).__init__()
        self.embed_Size = embed_Size
        self.embedding = nn.Embedding(vocab_size, embed_Size)
        self.pos_embedding = PositionalEncoding(embed_Size, dropout, max_len=max_len)
        self.blks = nn.Sequential()
        for i in range(n_layers):
            self.blks.add_module("d_block" + str(i),
                                 DecoderBlock(embed_Size, ffn_hidden, norm_shape, heads,
                                               dropout, mask))
        self.linear = nn.Linear(embed_Size, vocab_size)

    def forward(self, Y, X):
        Y = self.pos_embedding(self.embedding(Y) * math.sqrt(self.embed_Size))

        for layer in self.blks:
            Y = layer(Y, X)

        output = self.linear(Y)
        return output

class Transformerlayer(nn.Module):
    def __init__(self, vocab_size, embed_Size, ffn_hidden, max_len, heads, norm_shape,  n_layers, dropout, mask):
        super(Transformerlayer, self).__init__()
        self.encoder = Encoder(vocab_size, embed_Size, ffn_hidden, max_len, heads, norm_shape,  n_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_Size, ffn_hidden, max_len, heads, norm_shape, n_layers, dropout, mask)
  
    def forward(self, Y, X):
        X = self.encoder(X)
        Y = self.decoder(Y,X)
        return Y

decoder_blk = Transformerlayer(100, 24, 24, 100, 8, [3, 24], n_layers=2, dropout= 0.1, mask=None)
decoder_blk.eval()
X3 = torch.LongTensor([[2, 8, 2], [3, 1, 9]])
X4 = torch.LongTensor([[2, 8, 4], [4, 2, 0]])
out_put = decoder_blk(X3, X4)
print(out_put)
print(out_put.shape)
