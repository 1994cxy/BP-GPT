
from torch.utils.data import Dataset
import torch
import os
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

from .utiles_stim import get_stim
from .utile_resp import get_resp
from .get_roi import get_roi_index

class SDDataSetWR(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.sess_to_story = config.sess_to_story
        self.sessions = config.sessions
        self.win_length = config.win_length  # how much fmri time step in each sample
        self.data_dir = config.data_dir
        self.subject = f'UTS{str(config.subject).zfill(2)}'

        self.tokenizer = GPT2Tokenizer.from_pretrained("path to your gpt model")
        self.stories = self.load_story(train)
        self.word_seqs, word_length_info = get_stim(self.stories, config, stack=False)
        roi_index = get_roi_index(self.subject, roi_list=config.roi_list, full_head=False)
        self.resp, self.resp_all = get_resp(self.stories, config, stack=True, vox=roi_index)
        num_stim = sum([len(item) for _, item in self.word_seqs.items()]) if isinstance(self.word_seqs, dict) else len(self.word_seqs)
        num_resp = sum([item.shape[0] for _, item in self.resp.items()]) if isinstance(self.resp, dict) else len(self.resp)
        assert num_stim==num_resp, "number of stim is not equal to the resp" 
        self.data_num=num_stim


    def __getitem__(self, item):
        stim = self.word_seqs[item]
        stim_prefix = ' '.join(self.word_seqs[max(item-self.stim_prefix_length, 0): item])
        resp = self.resp[item]
        resp_all = self.resp_all[item]

        return (stim, stim_prefix, resp, tokens, mask, resp_all)

    def __len__(self):
        return self.data_num

    def load_story(self, train):
        stories = []
        with open(os.path.join(self.sess_to_story, "sess_to_story.json"), "r") as f:
            sess_to_story = json.load(f)
        if not train:
            return [sess_to_story[str('1')][1]]
        for sess in self.sessions:
            stories.extend(sess_to_story[str(sess)][0])
        return list(set(stories))
    
