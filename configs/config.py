
import argparse
import os

parser = argparse.ArgumentParser()


sub = 1
lr = 1e-4
win_length = 10
hop_length = win_length
finetune_GPT = 1
temper = 0.1 # 0.1
cuda = 5
brain_dim = 836  # sub1=836, sub2=2093, sub3=1303
prefix_length = 100
stage = 2  # 1 for text2text baseline, 2 for brain2text
use_beam_search = 0
spe_token = 1
Tmapper_drop = 0.1  # dropout ratio of transformer mapper
encoder_name = 'bert_mean'  # bert or bert_mean
exp_name = f"extend_data/B2T/finetune+spe+contras/prefix-{prefix_length}"

TD_path = "/data0/home/chenxiaoyu/code/auditory_decoding/semantic-decoding-gpt/checkpoints/extend_data/2024-07/T2T/finetune+spe/prefix-100/sub1/2024-07-23-16-09-00/fmri_prefix-TD-best.pt" # text decoder path

parser.add_argument("--stage", type=int, default=stage)
parser.add_argument("--lr", type=float, default=lr)

# semantic decoding data
DATA_TRAIN_DIR = "your dataset path"

parser.add_argument("--subject", nargs="+", type=int,
                    default=[sub])
parser.add_argument("--gpt", type=str, default="perceived")
parser.add_argument("--sessions", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5])
parser.add_argument("--sess_to_story", type=str, default=DATA_TRAIN_DIR)
parser.add_argument("--data_dir", type=str, default=DATA_TRAIN_DIR)
parser.add_argument("--win_length", type=int, default=win_length) 
parser.add_argument("--hop_length", type=int, default=hop_length)
parser.add_argument("--trim", type=int, default=5)
parser.add_argument("--only_info", type=int, default=0)
parser.add_argument("--spe_token", type=int, default=spe_token) 

# text model
parser.add_argument("--text_encoder_path", type=str)  # path to your BERT model
parser.add_argument("--encoder_dim", type=int, default=768) # 512 for clip, 768 for bert
parser.add_argument("--encoder_name", type=str, default=encoder_name)
parser.add_argument("--max_seq_length", type=int, default=128) 
parser.add_argument("--text_decoder_path", type=str, default=TD_path)  # must be given when use two stage traning


# prefix model
parser.add_argument('--out_dir', default=f'./checkpoints/' + exp_name + f'/sub{sub}')
parser.add_argument('--device', default=f'cuda:{cuda}')
parser.add_argument('--prefix_name', default='fmri_prefix', help='prefix for saved filenames')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_every', type=int, default=5)
parser.add_argument('--prefix_length', type=int, default=prefix_length)
parser.add_argument('--stim_prefix_length', type=int, default=5)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--finetune', type=int, default=finetune_GPT)
parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--is_rn', dest='is_rn', action='store_true')
parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
parser.add_argument('--gpt_path')  # path to your gpt model
parser.add_argument('--gpt_embedding_size', type=int, default=768)
parser.add_argument('--use_beam_search', type=int, default=use_beam_search)
parser.add_argument('--beam_size', type=int, default=200)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--Tmapper_drop', type=float, default=Tmapper_drop)

# brain model
parser.add_argument('--brain_dim', type=int, default=brain_dim)   # sub1=836, sub2=2093, sub3=1303
parser.add_argument('--BE_type', type=str, default='transformer', help='mlp/transformer')
parser.add_argument('--T2T_weight', type=int, default=0)
parser.add_argument('--B2T_weight', type=int, default=1)
parser.add_argument('--B2B_weight', type=int, default=0)
parser.add_argument('--C_weight', type=int, default=1)

parser.add_argument('--temper', type=int, default=temper)  # temperature of contrastive loss
parser.add_argument('--roi_list', type=list, default=['AC'])
parser.add_argument("--WR_path", type=str, default='/nfs/diskstation/DataStation/ChenXiaoyu/dataset/deep-fMRI-dataset-v2.2-split')
