
import torch
from torch.nn import Module
import torch.nn as nn

from .text_decoder import MLP, TransformerMapper, Transformer



class BrainDecoder(Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.input_dim = configs.brain_dim * configs.win_length
        self.prefix_length = configs.prefix_length
        self.gpt_embedding_size = configs.gpt_embedding_size

        if configs.mapping_type=='mlp':
            self.brain_project = MLP((self.gpt_embedding_size * self.prefix_length, (self.gpt_embedding_size * self.prefix_length) // 2, self.input_dim))
        elif configs.mapping_type=='transformer':
            self.brain_project = TransformerMapperD(self.gpt_embedding_size * self.prefix_length, self.gpt_embedding_size, self.configs.brain_dim,
                                                    self.prefix_length, self.configs.win_length)
        else:
            raise NotImplementedError(f"mapping_type:{configs.mapping_type} is not support yet!")

    def forward(self, brain_data):
        brain_prefix = self.brain_project(brain_data.view(-1, self.prefix_length * self.gpt_embedding_size)).view(-1, self.configs.win_length, self.configs.brain_dim)
        return brain_prefix
    
class TransformerMapperD(TransformerMapper):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.prefix_length, -1) # [batch_size, prefix_length, gpt_embedding_length]
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape) # [batch_size, brain_win_length, gpt_embedding_length]
        prefix = torch.cat((x, prefix), dim=1) 
        out = self.transformer(prefix)[:, self.prefix_length:]
        out = self.out_linear(out.view(out.shape[0], -1)).view(out.shape[0], self.brain_win_length, -1)
        return out

    def __init__(self, input_dim: int, dim_embedding: int, brain_dim:int, prefix_length: int, brain_win_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.brain_win_length = brain_win_length
        self.prefix_length = prefix_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(input_dim, prefix_length * dim_embedding)
        self.out_linear = nn.Linear(brain_win_length * dim_embedding, brain_win_length * brain_dim)
        self.prefix_const = nn.Parameter(torch.randn(brain_win_length, dim_embedding), requires_grad=True)