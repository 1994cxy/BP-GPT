
from typing import List, Optional, Tuple, Union
import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as nnf

from .text_decoder import MLP
from models.text_decoder import MLP, TransformerMapper



class BrainEncoder(Module):
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.brain_dim
        self.prefix_length = configs.prefix_length
        self.gpt_embedding_size = configs.gpt_embedding_size

        self.embedding_layer = BrainEmbeddings(configs.win_length, configs.brain_dim)
        self.BE_type = configs.BE_type
        if self.BE_type=='mlp':
            self.brain_project = MLP((self.input_dim, (self.gpt_embedding_size * self.prefix_length) // 2, self.gpt_embedding_size * self.prefix_length))
        elif self.BE_type=='transformer':
            self.brain_project = BrainTransformer(self.input_dim, self.gpt_embedding_size, self.prefix_length,
                                                                     clip_length=configs.win_length, num_layers=8, dropout=configs.Tmapper_drop)
        else:
            raise NotImplementedError(f"BE_type:{configs.BE_type} is not support yet!")


    def forward(self, brain_data):
        if self.BE_type=='mlp':
            brain_prefix = self.brain_project(brain_data.view(brain_data.shape[0], -1)).view(-1, self.prefix_length, self.gpt_embedding_size)
        else:
            brain_embedding = self.embedding_layer(brain_data)
            brain_prefix = self.brain_project(brain_embedding)
        return brain_prefix
    
class BrainEmbeddings(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        """
            max_position_embeddings:
            hidden_size:
        """
        super().__init__()
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))


    def forward(self, 
                input: torch.Tensor = None ,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values_length: int = 0,
                ) -> torch.Tensor:
        """
        """
        seq_length = input.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

      
        position_embeddings = self.position_embeddings(position_ids)
        embedding = input + position_embeddings

        return embedding


class BrainTransformer(nn.Module):

    def forward(self, x):
        x = self.linear(x)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, dropout: float = 0.):
        super(BrainTransformer, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers, dropout=dropout)
        self.linear = nn.Linear(dim_clip, dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False, dropout:float = 0.):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer, dropout=dropout))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer, dropout=dropout))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)



class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_position_embeddings", type=int, default=128)
    parser.add_argument("--hidden_size", type=float, default=836)
    
    config = parser.parse_args()

    embedding_model = BrainEmbeddings(config)

    fMRI = torch.zeros((32, 10, 836))
    embedding = embedding_model(fMRI)

