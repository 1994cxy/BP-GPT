
from torch.utils.data import Dataset
import torch
import os
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import h5py
import numpy as np

from .utiles_stim import get_stim
from .utile_resp import get_resp
from .get_roi import get_roi_index

class SDDataSet(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.sess_to_story = config.sess_to_story
        self.sessions = config.sessions
        self.win_length = config.win_length  # how much fmri time step in each sample
        self.data_dir = config.data_dir
        self.subject = config.subject
        self.stim_prefix_length = config.stim_prefix_length
        self.only_info = config.only_info

        self.tokenizer = GPT2Tokenizer.from_pretrained("path to your gpt model")

        self.data = []
        for i in self.subject:
            sub = f'UTS{str(i).zfill(2)}'
            stories = self.load_story(train, sub)
            word_seqs_, word_length_info_ = get_stim(stories, config, stack=False, spe_token=config.spe_token)
            if config.roi_list:
                self.roi_index = get_roi_index(sub, roi_list=config.roi_list, full_head=False)
            else:
                self.roi_index = None
            resp_ = get_resp(sub, stories, config, stack=True, vox=self.roi_index, only_info=self.only_info)
            num_stim = sum([len(item) for _, item in word_seqs_.items()]) if isinstance(word_seqs_, dict) else len(word_seqs_)
            num_resp = sum([item.shape[0] for _, item in resp_.items()]) if isinstance(resp_, dict) else len(resp_)
            assert num_stim==num_resp, f"number of stim is not equal to the resp of subject {sub}" 
            self.data.extend(list(zip(word_seqs_, resp_)))
        
        self.data_num=len(self.data)


    def __getitem__(self, item):
        if self.only_info: 
            stim, resp_info = self.data[item]
            resp_path, start, end = resp_info
            with h5py.File(resp_path, "r") as hf:
                if self.roi_index:
                    resp = np.nan_to_num(hf["data"][int(start): int(end), self.roi_index])
                else:
                    resp = np.nan_to_num(hf["data"][int(start): int(end)])
        
        else:
            stim, resp = self.data[item]

        tokens, mask = self.get_token_mask(stim, max_seq_len=self.config.max_seq_length, prefix_length=self.config.prefix_length)

        return (stim, resp, tokens, mask)

    def __len__(self):
        return self.data_num

    def load_story(self, train, sub):
        stories = []
        with open(os.path.join(self.sess_to_story, "sess_to_story.json"), "r") as f:
            sess_to_story = json.load(f)
        if not train:
            return [sess_to_story[str('1')][1]]
        for sess in self.sessions:
            stories.extend(sess_to_story[str(sess)][0])

        if sub in ['UTS01', 'UTS02', 'UTS03'] and train:  
            extend_stories = ['adollshouse', 'adventuresinsayingyes', 'afatherscover', 'againstthewind', 'alternateithicatom', 'avatar', 'backsideofthestorm', 'becomingindian', 'beneaththemushroomcloud', 'birthofanation', 'bluehope', 'breakingupintheageofgoogle', 'buck', 'catfishingstrangerstofindmyself', 'cautioneating', 'christmas1940', 'cocoonoflove', 'comingofageondeathrow', 'exorcism', 'eyespy', 'firetestforlove', 'food', 'forgettingfear', 'fromboyhoodtofatherhood', 'gangstersandcookies', 'goingthelibertyway', 'goldiethegoldfish', 'golfclubbing', 'gpsformylostidentity', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'ifthishaircouldtalk', 'inamoment', 'itsabox', 'jugglingandjesus', 'kiksuya', 'leavingbaghdad', 'legacy', 'life', 'lifeanddeathontheoregontrail', 'lifereimagined', 'listo', 'mayorofthefreaks', 'metsmagic', 'mybackseatviewofagreatromance', 'myfathershands', 'myfirstdaywiththeyankees', 'naked', 'notontheusualtour', 'odetostepfather', 'onlyonewaytofindout', 'penpal', 'quietfire', 'reachingoutbetweenthebars', 'shoppinginchina', 'singlewomanseekingmanwich', 'sloth', 'souls', 'stagefright', 'stumblinginthedark', 'superheroesjustforeachother', 'sweetaspie', 'swimmingwithastronauts', 'thatthingonmyarm', 'theadvancedbeginner', 'theclosetthatateeverything', 'thecurse', 'thefreedomridersandme', 'theinterview', 'thepostmanalwayscalls', 'theshower', 'thetiniestbouquet', 'thetriangleshirtwaistconnection', 'threemonths', 'thumbsup', 'tildeath', 'treasureisland', 'undertheinfluence', 'vixenandtheussr', 'waitingtogo', 'whenmothersbullyback', 'wildwomenanddancingqueens']
            stories.extend(extend_stories)
        return list(set(stories))
    
    def get_token_mask(self, stim, max_seq_len=77, prefix_length=10):
        tokens_stim = torch.tensor(self.tokenizer.encode(stim), dtype=torch.int64)
        padding = max_seq_len - tokens_stim.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens_stim, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            print(f'stim token length is lager than {max_seq_len}!')
            tokens = tokens_stim[:max_seq_len]
        mask = tokens.ge(0) 
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def get_token_mask_stim_pre(self, stim, stim_prefix, max_seq_len=77, prefix_length=10):
        tokens_stim = torch.tensor(self.tokenizer.encode(stim), dtype=torch.int64)
        tokens_stim_prefix = torch.tensor(self.tokenizer.encode(stim_prefix), dtype=torch.int64)
       
        padding = max_seq_len - (tokens_stim.shape[0] + tokens_stim_prefix.shape[0])
        if padding > 0:
            tokens = torch.cat((tokens_stim_prefix, tokens_stim))
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = torch.cat((tokens_stim_prefix, tokens_stim))
            tokens = tokens[abs(padding):]
     
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(prefix_length), mask), dim=0)  # adding prefix mask  
        return tokens, mask

