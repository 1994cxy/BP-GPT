
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
# from google.colab import files
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu

from models.text_decoder import CldTextDecoder
from dataset import SDDataSet
from models.text_encoder import CldTextEncoder
from configs import parser
from models.brain_encoder import BrainEncoder
from utils import WER, BERTSCORE

cal_wer = WER(use_score = False)
cal_bert_score = BERTSCORE(rescale = False, score = "recall")

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
    # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
    sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
        mask_joint_mod
    ).view(2 * batch_size, -1)

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

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.', stim_prefix=None):

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
            if stim_prefix:
                stim_prefix_tokens = torch.tensor(tokenizer.encode(stim_prefix))
                padding = 77 - stim_prefix_tokens.shape[0]
                if padding > 0:
                    stim_prefix_tokens = torch.cat((stim_prefix_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    stim_prefix_tokens = tokens[abs(padding):]
                stim_prefix_tokens = stim_prefix_tokens.unsqueeze(0).to(device)
                stim_prefix_tokens = model.gpt.transformer.wte(stim_prefix_tokens)
                generated = torch.cat((generated, stim_prefix_tokens), dim=1)
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


def generate2(
        model,
        configs,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
        spe_token: str='$',
        win_length: int=10
):
    model.eval()
    generated_num = 0
    generated_list = []
    # use special token as stop token
    if configs.spe_token:
        stop_token_index = tokenizer.encode(spe_token)[0]
    else:
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

                if configs.spe_token:
                    if stop_token_index == next_token.item():
                        win_length -= 1
                        if win_length<=1:
                            break
            
            tokens = tokens.squeeze().cpu().numpy()
            if not tokens.shape:
                output_list = [tokens]
            else:
                output_list = list(tokens)
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def cal_score(gt_sentence, generated_sentence):
    meteor = meteor_score.meteor_score([gt_sentence.split()], generated_sentence.split())
    BLEU1 = sentence_bleu([gt_sentence.split()], generated_sentence.split(), weights=[1, 0, 0, 0])
    bert_score = cal_bert_score.score([gt_sentence], [generated_sentence])
    wer_score = cal_wer.score([gt_sentence], [generated_sentence])

    return (wer_score, BLEU1, meteor, bert_score)

def cal_score_segment(gt_sentence, generated_sentence):
    meteor = meteor_score.meteor_score([gt_sentence.split()], generated_sentence.split())
    BLEU1 = sentence_bleu([gt_sentence.split()], generated_sentence.split(), weights=[1, 0, 0, 0])
    bert_score = cal_bert_score.score([gt_sentence], [generated_sentence])
    wer_score = cal_wer.score([gt_sentence], [generated_sentence])

    return wer_score, BLEU1, meteor, bert_score

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def predict_word_rate(resp, wt, vox, mean_rate):
    """predict word rate at each acquisition time
    """
    delresp = make_delayed(np.expand_dims(resp[vox], 0), [-4, -3, -2, -1])
    rate = ((delresp.dot(wt) + mean_rate)).reshape(-1).clip(min = 0)
    return np.round(rate).astype(int)[0]


def run_test(text_encoder, text_decoder, brain_encoder, test_set, configs):

    is_gpu = True #@param {type:"boolean"}
    device = torch.device(configs.device) if is_gpu else "cpu"

    text_encoder.to(device)
    text_encoder.eval()

    text_decoder.to(device)
    text_decoder.eval()

    if configs.stage==2: #brain2text
        brain_encoder.to(device)
        brain_encoder.eval()

    use_beam_search = configs.use_beam_search #@param {type:"boolean"}
    
   
    ori_text, B2T_text, T2T_text = '', '', ''


    B2T_meteor_segment, B2T_BLEU1_segment, B2T_wer_segment, B2T_bert_score_segment = [], [], [], []
    T2T_meteor_segment, T2T_BLEU1_segment, T2T_wer_segment, T2T_bert_score_segment = [], [], [], []
    # print(len(test_set))

    WR_ = np.load(os.path.join(configs.WR_path, f'WR-UTS{str(configs.subject[0]).zfill(2)}-win{configs.win_length}-hop{configs.hop_length}.npy'))

    decode_text = []
    gpt_loss_text2text = torch.tensor(0.).to(configs.device)
    gpt_loss_brain2text = torch.tensor(0.).to(configs.device)
    contrastive_loss = torch.tensor(0.).to(configs.device)
    for step, data in enumerate(test_set):
        assert configs.win_length>=configs.hop_length and configs.win_length%configs.hop_length==0
        if step%(configs.win_length//configs.hop_length)!=0: 
            continue


        stim_spe, resp, tokens, mask = data
        tokens = tokens.to(configs.device)
        mask = mask.to(configs.device)

        B2T_seg, T2T_seg = '', ''
        
        # word rate
        text_length = int(WR_[step])
        if configs.spe_token:
            text_length = 128 # max text length of text encoder
        else:
            text_length = WR_ * configs.win_length 
        
        if configs.spe_token:
            stim = stim_spe.replace('$', '').replace('=', '')
        else:
            stim = stim_spe
        ori_text+=' '+stim

        # text encode and decode
        with torch.no_grad():
            bert_embedding, text_input_ids = text_encoder([stim_spe], device=configs.device)
            prefix_text = text_decoder.model.linear(bert_embedding) 
            prefix_embed = text_decoder.model.clip_project(prefix_text).view(-1, text_decoder.model.prefix_length, text_decoder.model.gpt_embedding_size)
            if use_beam_search:
                generated_text_prefix = generate_beam(text_decoder.model, text_decoder.tokenizer, beam_size=configs.beam_size, embed=prefix_embed, entry_length=text_length, stim_prefix=None)[0]
            else:
                generated_text_prefix = generate2(text_decoder.model, configs, text_decoder.tokenizer, embed=prefix_embed, entry_length=text_length)
            # eval loss
            out_text2text, prefix_text = text_decoder.train_forward(tokens.unsqueeze(0), bert_embedding, mask)
            logits_text2text = out_text2text.logits[:, configs.prefix_length - 1: -1]
            gpt_loss_text2text += nnf.cross_entropy(logits_text2text.reshape(-1, logits_text2text.shape[-1]), tokens.flatten(), ignore_index=0)
            del out_text2text, prefix_text
        generated_text_prefix = generated_text_prefix.replace('$', "")
        generated_text_prefix = generated_text_prefix.replace('=', "")
        T2T_text+=' '+generated_text_prefix
        T2T_seg += ' '+generated_text_prefix

        
        if configs.stage==2: #brain2text
            # decode brain
            with torch.no_grad():
                prefix_embed = brain_encoder(torch.tensor(resp).float().to(device).unsqueeze(0))
                if use_beam_search:
                    generated_text_prefix = generate_beam(text_decoder.model, text_decoder.tokenizer, beam_size=configs.beam_size, embed=prefix_embed, entry_length=text_length, stim_prefix=None)[0]
                else:
                    generated_text_prefix = generate2(text_decoder.model, configs, text_decoder.tokenizer, embed=prefix_embed, entry_length=text_length)

            # eval loss
            out_brain2text = text_decoder.brain_forward(tokens.unsqueeze(0), prefix_embed, mask)
            logits_brain2text = out_brain2text.logits[:, configs.prefix_length - 1: -1]
            gpt_loss_brain2text += nnf.cross_entropy(logits_brain2text.reshape(-1, logits_brain2text.shape[-1]), tokens.flatten(), ignore_index=0)

            contrastive_loss += cal_contrastive_loss(prefix_embed.view(prefix_embed.shape[0], -1), prefix_text.view(prefix_text.shape[0], -1).detach(), configs.temper)
            del out_brain2text

            generated_text_prefix = generated_text_prefix.replace('$', "")
            generated_text_prefix = generated_text_prefix.replace('=', "")
            B2T_text+=' '+generated_text_prefix
            B2T_seg += ' '+generated_text_prefix

        
        # brain 2 text score in win length
        wer_score_, BLEU1_, meteor_, bert_score_ = cal_score_segment(stim, B2T_seg)
        B2T_wer_segment.append(wer_score_)
        B2T_BLEU1_segment.append(BLEU1_)
        B2T_meteor_segment.append(meteor_)
        B2T_bert_score_segment.append(bert_score_)

        decode_text.append((BLEU1_, meteor_, B2T_seg, stim))

        # text 2 text score in win length
        wer_score_, BLEU1_, meteor_, bert_score_ = cal_score_segment(stim, T2T_seg)
        T2T_wer_segment.append(wer_score_)
        T2T_BLEU1_segment.append(BLEU1_)
        T2T_meteor_segment.append(meteor_)
        T2T_bert_score_segment.append(bert_score_)

    # B2T score
    # B2T_wer_score, B2T_BLEU1, B2T_meteor, B2T_bert_score = cal_score(ori_text, B2T_text)
    B2T_wer_segment = sum(B2T_wer_segment)/len(B2T_wer_segment)
    B2T_BLEU1_segment = sum(B2T_BLEU1_segment)/len(B2T_BLEU1_segment)
    B2T_meteor_segment = sum(B2T_meteor_segment)/len(B2T_meteor_segment)
    B2T_bert_score_segment = sum(B2T_bert_score_segment)/len(B2T_bert_score_segment)

    # T2T score
    # T2T_wer_score, T2T_BLEU1, T2T_meteor, T2T_bert_score = cal_score(ori_text, T2T_text)
    T2T_wer_segment = sum(T2T_wer_segment)/len(T2T_wer_segment)
    T2T_BLEU1_segment = sum(T2T_BLEU1_segment)/len(T2T_BLEU1_segment)
    T2T_meteor_segment = sum(T2T_meteor_segment)/len(T2T_meteor_segment)
    T2T_bert_score_segment = sum(T2T_bert_score_segment)/len(T2T_bert_score_segment)

    B2T_score = {
        'seg_wer_score': B2T_wer_segment,
        'seg_BLEU1': B2T_BLEU1_segment,
        'seg_meteor': B2T_meteor_segment,
        'seg_bert_score': B2T_bert_score_segment,
    }

    T2T_score = {
        'seg_wer_score': T2T_wer_segment,
        'seg_BLEU1': T2T_BLEU1_segment,
        'seg_meteor': T2T_meteor_segment,
        'seg_bert_score': T2T_bert_score_segment,
    }

    # loss
    loss = {
        'text2text': gpt_loss_text2text/len(test_set),
        'brain2text': gpt_loss_brain2text/len(test_set),
        'constrastive': contrastive_loss/len(test_set)
    }
    # print(B2T_score)
    return (loss, B2T_score, T2T_score, ori_text, B2T_text, T2T_text)



if __name__=='__main__':
    configs = parser.parse_args()
    test_set = SDDataSet(configs, train=False)

    text_encoder = CldTextEncoder(configs, configs.text_encoder_path)

    save_path = 'check point path'
    text_model_path = os.path.join(save_path, f'fmri_prefix_TD_last.pt')
    text_decoder = CldTextDecoder(model_path=None, gpt_path=configs.gpt_path, 
                                      prefix_length=configs.prefix_length, encoder_dim=configs.encoder_dim,
                                      latent_dim = configs.latent_dim, mapping_type=configs.mapping_type, 
                                      finetune=configs.finetune, time_length=1, dropout=configs.Tmapper_drop)
                
    text_decoder.load_state_dict(torch.load(text_model_path, map_location="cpu"))

    # brain_encoder = None
    brain_encoder = BrainEncoder(configs)
    brain_model_path = os.path.join(save_path, f'fmri_prefix-BE-last.pt')
    brain_encoder.load_state_dict(torch.load(brain_model_path, map_location="cpu"))

    # word rate model
    word_rate_voxels = "auditory"
    load_location = 'word rate model path'
    # load_location = f"~/semantic-decoding-main/prefit_models/S{configs.subject[0]}"
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)

    configs.subject = 1
    configs.brain_dim = 836 # sub1=836, sub2=2093, sub3=1303
    configs.spe_token = 1

    run_test(text_encoder, text_decoder, brain_encoder, test_set, configs)

