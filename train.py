
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from configs import parser
from dataset import SDDataSet
from models.text_encoder import CldTextEncoder
from models.text_decoder import CldTextDecoder
from models.brain_encoder import BrainEncoder
from models.brain_decoder import BrainDecoder
from inference import run_test

def cal_contrastive_loss(brain_prefix, text_prefix, temperature=0.05):
    batch_size = len(brain_prefix)

    # Negative pairs: everything that is not in the current joint-modality pair
    out_joint_mod = torch.cat(
        [text_prefix, brain_prefix], dim=0
    )
    # [2*B, 2*B]
    inner_product = torch.mm(out_joint_mod, out_joint_mod.t().contiguous())

    norm_mask = (torch.zeros_like(inner_product)
            + torch.eye(2 * batch_size, device=inner_product.device)
    ).bool()
    norm = inner_product.masked_select(
        norm_mask
    ).view(2 * batch_size, -1).repeat(1, 2 * batch_size)

    inner_product = inner_product/norm

    sim_matrix_joint_mod = torch.exp(
        inner_product / temperature
    )
    # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
    mask_joint_mod = (
            torch.ones_like(sim_matrix_joint_mod)
            - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
    ).bool()

    # Mask out the text-text pair from the negative pair
    mask_joint_mod[:batch_size, :batch_size]=False

    # # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
    # sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
    #     mask_joint_mod
    # ).view(2 * batch_size, -1)
    sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
        mask_joint_mod
    )

    # Positive pairs: cosine loss joint-modality
    inner_product = torch.sum(text_prefix * brain_prefix, dim=-1)
    norm = torch.sum(text_prefix * text_prefix, dim=-1)
    inner_product = inner_product/norm

    pos_sim_joint_mod = torch.exp(
        inner_product / temperature
    ).sum()

    loss_joint_mod = -torch.log(
        pos_sim_joint_mod / sim_matrix_joint_mod.sum()
    )

    return loss_joint_mod

def train(dataset, text_encoder, text_decoder, brain_encoder, brain_decoder, configs,
          output_dir: str = ".", output_prefix: str = ""):
    device = configs.device
    lr = configs.lr
    batch_size = configs.bs
    epochs = configs.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(output_dir)

    writer.add_text('hyper', str(configs), 0)


    text_encoder.to(device)
    text_encoder.eval()
    text_decoder.to(device)
    if configs.stage==1: #for text2text training
        text_decoder.train()
    else:  #for brain2text training
        if configs.finetune:
            text_decoder.train()
        else:
            text_decoder.eval()

        brain_encoder.to(device)
        brain_encoder.train()
        if configs.B2B_weight!=0: 
            brain_decoder.to(device)
            brain_decoder.train()


    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_set = SDDataSet(configs, train=False)

    warmup_steps = 5*len(train_dataloader)

    if configs.stage==1: #for text2text training
        optimizer_TD = AdamW(text_decoder.parameters(), lr=lr)
        scheduler_TD = get_linear_schedule_with_warmup(
            optimizer_TD, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
        )

    else: #for brain2text training

        optimizer_BE = AdamW(brain_encoder.parameters(), lr=lr)
        scheduler_BE = get_linear_schedule_with_warmup(
            optimizer_BE, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
        )
        if configs.B2B_weight!=0: 
            optimizer_BD = AdamW(brain_decoder.parameters(), lr=lr)
            scheduler_BD = get_linear_schedule_with_warmup(
                optimizer_BD, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
            )

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        

        if configs.stage==1: 
            text_decoder.train()
        else:
            if configs.finetune:
                text_decoder.train()
            else:
                text_decoder.eval()
            brain_encoder.train()
            if configs.B2B_weight!=0: 
                brain_decoder.train()

       
        for idx, ((stim, resp, tokens, mask)) in enumerate(train_dataloader):
            # text encode
            tokens = tokens.to(device)
            prefix_text, text_input_ids = text_encoder(list(stim), device=device)

            if configs.stage==1: 
                # text decode and text recon loss
                out_text2text, prefix_text = text_decoder.train_forward(tokens, prefix_text, mask)
                logits_text2text = out_text2text.logits[:, configs.prefix_length - 1: -1]
                gpt_loss_text2text = nnf.cross_entropy(logits_text2text.reshape(-1, logits_text2text.shape[-1]), tokens.flatten(), ignore_index=0)

                total_loss = configs.T2T_weight * gpt_loss_text2text
                gpt_loss_brain2text, brain_recon_loss, contrastive_loss = 0, 0, 0
            else:
                # brain encode and decode, and recon loss
                resp = resp.to(device).float()
                brain_prefix = brain_encoder(resp)
                if configs.B2B_weight!=0: 
                    brain_recon = brain_decoder(brain_prefix)
                    brain_recon_loss = nnf.mse_loss(brain_recon, resp)
                else:
                    brain_recon_loss = 0

                # run text decoder to get prefix
                out_text2text, prefix_text = text_decoder.train_forward(tokens, prefix_text, mask)

                # contrastive loss between text_prefix and brain_prefix
                contrastive_loss = cal_contrastive_loss(brain_prefix.view(brain_prefix.shape[0], -1), prefix_text.view(prefix_text.shape[0], -1).detach(), configs.temper)

                # brain2text decode
                out_brain2text = text_decoder.brain_forward(tokens, brain_prefix, mask)
                logits_brain2text = out_brain2text.logits[:, configs.prefix_length - 1: -1]
                gpt_loss_brain2text = nnf.cross_entropy(logits_brain2text.reshape(-1, logits_brain2text.shape[-1]), tokens.flatten(), ignore_index=0)

                total_loss = configs.B2T_weight * gpt_loss_brain2text + configs.B2B_weight * brain_recon_loss + configs.C_weight * contrastive_loss
                gpt_loss_text2text = 0
            total_loss.backward()

            writer.add_scalar('train/brain2text_loss', gpt_loss_brain2text, epoch*len(train_dataloader) + idx)
            writer.add_scalar('train/text2text_loss', gpt_loss_text2text, epoch*len(train_dataloader) + idx)
            writer.add_scalar('train/brain_recon_loss', brain_recon_loss, epoch*len(train_dataloader) + idx)
            writer.add_scalar('train/contrastive_loss', contrastive_loss, epoch*len(train_dataloader) + idx)
            writer.add_scalar('train/total_loss', total_loss, epoch*len(train_dataloader) + idx)

            if configs.stage==1:  
                optimizer_TD.step()
                scheduler_TD.step()
                optimizer_TD.zero_grad()
            else: 
                optimizer_BE.step()
                scheduler_BE.step()

                if configs.B2B_weight!=0: 
                    optimizer_BD.step()
                    scheduler_BD.step()
                    optimizer_BD.zero_grad()

                optimizer_BE.zero_grad()
            
            progress.set_postfix({"loss": total_loss.item()})
            progress.update()

        progress.close()

    # eval
    loss, B2T_score, T2T_score, ori_text, B2T_text, T2T_text = run_test(text_encoder, text_decoder, brain_encoder, test_set, configs)

    writer.add_scalars('eval_loss/text2text', {'text2text': loss['text2text']}, epoch)
    writer.add_scalars('eval_loss/brain2text', {'brain2text': loss['brain2text']}, epoch)
    writer.add_scalars('eval_loss/constrastive', {'constrastive': loss['constrastive']}, epoch)
    writer.add_scalars('eval/WER_segment', {'B2T': B2T_score['seg_wer_score'], 'T2T': T2T_score['seg_wer_score']}, epoch)
    writer.add_scalars('eval/METEOR_segment', {'B2T': B2T_score['seg_meteor'], 'T2T': T2T_score['seg_meteor']}, epoch)
    writer.add_scalars('eval/BLEU1_segment', {'B2T': B2T_score['seg_BLEU1'], 'T2T': T2T_score['seg_BLEU1']}, epoch)
    writer.add_scalars('eval/BERTSCORE_segment', {'B2T': B2T_score['seg_bert_score'], 'T2T': T2T_score['seg_bert_score']}, epoch)

    if epoch==0:
        writer.add_text('ori_text', ori_text, epoch)
    writer.add_text('B2T_text', B2T_text, epoch)
    writer.add_text('T2T_text', T2T_text, epoch)
    
    # save the last model
    if configs.stage==1:  
        torch.save(
            text_decoder.state_dict(),
            os.path.join(output_dir, f"{output_prefix}_TD_last.pt"),
        )
    else:
        torch.save(
        brain_encoder.state_dict(),
        os.path.join(output_dir, f"{output_prefix}_BE_last.pt"),
        )
        torch.save(
            text_decoder.state_dict(),
            os.path.join(output_dir, f"{output_prefix}_TD_last.pt"),
        )
        if configs.B2B_weight!=0: 
            torch.save(
            brain_decoder.state_dict(),
            os.path.join(output_dir, f"{output_prefix}_BD_last.pt"),
            )


    return text_decoder


if __name__=='__main__':
    configs = parser.parse_args()
    train_set = SDDataSet(configs)
    
    text_encoder = CldTextEncoder(configs, configs.text_encoder_path)

    if configs.stage == 1: 
        brain_encoder = None
        brain_decoder = None
    else:
        brain_encoder = BrainEncoder(configs)
        if configs.B2B_weight==0:
            brain_decoder = None
        else:
            brain_decoder = BrainDecoder(configs)
    if not configs.finetune:
        text_decoder = CldTextDecoder(model_path=None, gpt_path=configs.gpt_path, 
                                      prefix_length=configs.prefix_length, encoder_dim=configs.encoder_dim,
                                      latent_dim = configs.latent_dim, mapping_type=configs.mapping_type, 
                                      finetune=configs.finetune, time_length=1, dropout=configs.Tmapper_drop)

        if configs.stage==2: 
            state_dict = torch.load(configs.text_decoder_path, map_location=configs.device)
            text_decoder.load_state_dict(state_dict)

        print("Train only prefix")
    else:
        text_decoder = CldTextDecoder(model_path=None, gpt_path=configs.gpt_path, 
                                      prefix_length=configs.prefix_length, encoder_dim=configs.encoder_dim,
                                      latent_dim = configs.latent_dim, mapping_type=configs.mapping_type, 
                                      finetune=configs.finetune, time_length=1, dropout=configs.Tmapper_drop)

        if configs.stage==2: 
            state_dict = torch.load(configs.text_decoder_path, map_location=configs.device)
            text_decoder.load_state_dict(state_dict)

        print("Train prefix and finetune GPT!")
    
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(configs.out_dir, formatted_date)
    
    train(train_set, text_encoder, text_decoder, brain_encoder, brain_decoder, configs, output_dir=out_dir, output_prefix=configs.prefix_name)