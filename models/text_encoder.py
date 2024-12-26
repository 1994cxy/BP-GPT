
import os
from typing import List, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

class CldTextEncoder(nn.Module):
    def __init__(
        self,
        configs,
        model_path: str,
        finetune: bool = False,
        last_hidden_state: bool = False,
    ) -> None:

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModel.from_pretrained(model_path)

        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        if "clip" in model_path:
            self.max_length = 77
            self.text_encoded_dim = self.text_model.config.text_config.hidden_size
            if last_hidden_state:
                self.name = "clip_hidden"
            else:
                self.name = "clip"
        elif configs.encoder_name=='bert_mean':
            self.name = "bert_mean"
            self.max_length = 512
            self.text_encoded_dim = self.text_model.config.hidden_size
        elif configs.encoder_name=='bert':
            self.name = "bert"
            self.max_length = 512
            self.text_encoded_dim = self.text_model.config.hidden_size
        else:
            raise ValueError(f"Model {model_path} not supported")

    def forward(self, texts: List[str], device):
        text_input_ids = None
        # get prompt text embeddings
        if self.name in ["clip", "clip_hidden"]:
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            # split into max length Clip can handle
            if text_input_ids.shape[-1] > self.max_length:
                text_input_ids = text_input_ids[:, :self.max_length]
        elif "bert" in self.name :
            text_inputs = self.tokenizer(texts,
                                         return_tensors="pt",
                                         padding=True)

        # use pooled ouuput if latent dim is two-dimensional
        # pooled = 0 if self.latent_dim[0] == 1 else 1 # (bs, seq_len, text_encoded_dim) -> (bs, text_encoded_dim)
        # text encoder forward, clip must use get_text_features
        if self.name == "clip":
            # (batch_Size, text_encoded_dim)
            text_embeddings = self.text_model.get_text_features(
                text_input_ids.to(self.text_model.device))
            # (batch_Size, 1, text_encoded_dim)
            text_embeddings = text_embeddings.unsqueeze(1)
        elif self.name == "clip_hidden":
            # (batch_Size, seq_length , text_encoded_dim)
            text_embeddings = self.text_model.text_model(
                text_input_ids.to(self.text_model.device)).last_hidden_state
        elif self.name == "bert":
            # (batch_Size, seq_length , text_encoded_dim) => (batch_Size, 1 , text_encoded_dim)
         
            text_embeddings = self.text_model(
                **text_inputs.to(self.text_model.device)).last_hidden_state
            end_idx = []
            for i in range(text_embeddings.shape[0]):
                end_idx.append(torch.nonzero(text_inputs.input_ids[i])[-1][0].item())
            text_embeddings = text_embeddings[torch.arange(text_embeddings.shape[0]), torch.tensor(end_idx)]
            text_embeddings = text_embeddings.unsqueeze(1)
        elif self.name == "bert_mean":
            text_embeddings = self.text_model(
                **text_inputs.to(self.text_model.device)).last_hidden_state
            text_embeddings = text_embeddings.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError(f"Model {self.name} not implemented")

        return text_embeddings, text_input_ids.to(self.text_model.device) if text_input_ids else None  # clip embdding, token


