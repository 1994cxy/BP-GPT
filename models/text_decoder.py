
from typing import List, Union, Optional, Tuple
import torch.nn as nn
from torch.nn import functional as nnf
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from enum import Enum
import numpy as np
from itertools import chain


class CldTextDecoder(nn.Module):

    def __init__(
        self,
        model_path: str,
        gpt_path: str,
        mapping_type = str,
        finetune: bool = False,
        last_hidden_state: bool = False,
        use_beam_search: bool = False,
        latent_dim: int = 128,
        prefix_length: int = 10,
        encoder_dim:int = 512,
        time_length: int=1,
        dropout: float=0.
    ) -> None:

        super().__init__()

        self.use_beam_search = use_beam_search
        self.latent_dim = latent_dim

       
        self.prefix_length = prefix_length
        if finetune:
            self.model = ClipCaptionModel(prefix_length=self.prefix_length, prefix_size=self.latent_dim, encoder_dim=encoder_dim, mapping_type=mapping_type, time_length=time_length, dropout=dropout)
        else:
            self.model = ClipCaptionPrefix(prefix_length=self.prefix_length, prefix_size=self.latent_dim, encoder_dim=encoder_dim, mapping_type=mapping_type, time_length=time_length, dropout=dropout)


        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_path)

    def forward(self, latent):
        """
        """
        latent = self.model.linear(latent.float())

        # get prompt text embeddings
        prefix_embed = self.model.clip_project(latent).reshape(latent.shape[0], self.prefix_length, -1)

        generated_text = []
        for embed in prefix_embed:
            if self.use_beam_search:
                generated_text.append(self.generate_beam(self.model, self.tokenizer, embed=embed.unsqueeze(0))[0])
            else:
                generated_text.append(self.generate2(self.model, self.tokenizer, embed=embed.unsqueeze(0)))
        return generated_text

    def train_forward(self, tokens, prefix, mask=None, labels=None):
        """
        """
        embedding_text = self.model.gpt.transformer.wte(tokens)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # print(prefix.shape)
        prefix = self.model.linear(prefix) 
        prefix_projections = self.model.clip_project(prefix).view(-1, self.model.prefix_length, self.model.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.model.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.model.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask.to(self.model.gpt.device))
        return out, prefix_projections

    def brain_forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.model.gpt.transformer.wte(tokens)
        
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.model.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.model.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask.to(self.model.gpt.device))
        return out

    def generate2(
            self,
            model,
            tokenizer,
            tokens=None,
            prompt=None,
            embed=None,
            entry_count=1,
            entry_length=67,  # maximum number of words
            top_p=0.8,
            temperature=1.,
            stop_token: str = '.',
    ):
        model.eval()
        generated_num = 0
        generated_list = []
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        device = next(model.parameters()).device

        with torch.no_grad():

            for entry_idx in trange(entry_count):
                if embed is not None:
                    generated = embed
                else:
                    if tokens is None:
                        tokens = torch.tensor(tokenizer.encode(prompt))
                        tokens = tokens.unsqueeze(0).to(device)

                    generated = model.gpt.transformer.wte(tokens)

                for i in range(entry_length):

                    outputs = model.gpt(inputs_embeds=generated)
                    logits = outputs.logits
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                        ..., :-1
                                                        ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                    next_token_embed = model.gpt.transformer.wte(next_token)
                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    if stop_token_index == next_token.item():
                        break
                if tokens.shape[-1]==1:  
                    output_list=[tokens.squeeze().cpu().numpy()]
                else:
                    output_list = list(tokens.squeeze().cpu().numpy())
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)

        return generated_list[0]

    def generate_beam(self, model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                      entry_length=67, temperature=1., stop_token: str = '.'):

        model.eval()
        stop_token_index = tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = model.gpt.transformer.wte(tokens)
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts

    def get_mask(self, tokens):
        padding = 77 - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:77]
       
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length).to(mask.device), mask), dim=0)  # adding prefix mask

        return mask


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix = self.linear(prefix)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, time_length: int = 1, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: str = 'mlp', encoder_dim: int=512, dropout=0.):
        """
        """
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("path to your gpt model")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.linear = nn.Linear(encoder_dim, prefix_size)  
        if mapping_type == 'mlp':
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     time_length, num_layers, dropout=dropout)
            


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return chain(self.clip_project.parameters(), self.linear.parameters())

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, dropout=0.):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers, dropout=dropout)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
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
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False,
                 dropout=0.):
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
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer, dropout=dropout))
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