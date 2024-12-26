
import os
import numpy as np
import json
from typing import Iterable

from .utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from .utils_ridge.dsutils import make_word_ds
from .utils_ridge.interpdata import lanczosinterp2D
from .utils_ridge.util import make_delayed

def get_story_wordseqs(stories, config):
    """loads words and word times of stimulus stories
    """
    grids = load_textgrids(stories, config.data_dir)
    with open(os.path.join(config.data_dir, "respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs

def get_stim(stories, config, stack=False, spe_token=False):
    """extract quantitative features of stimulus stories
    """
    word_seqs = get_story_wordseqs(stories, config)  # word_seqs=[story_num, word_num]
    ds_word, word_length_info = interp_word(stories, word_seqs, config)  # ds_word=[story_num, fmri_num]
    ds_WR = {}
    for story in stories:
        word_list = []
        win_num = len(ds_word[story])//config.hop_length
        win_num -= config.win_length//config.hop_length - 1  
        
        for i in range(win_num):
            start = i*config.hop_length
            end = i*config.hop_length + config.win_length
            if spe_token:
                sentence = ' $ '.join(ds_word[story][start: end])
                sentence = '= '+sentence+' $'
                word_list.append(sentence)
            else:
                word_list.append(' '.join(ds_word[story][start: end]))
        ds_word[story] = word_list
        WR_ = [len(word_list[i].split(' ')) for i in range(len(word_list))]
        ds_WR[story] = [sum(WR_)/len(WR_)] * len(word_list)
    
    if stack: return np.vstack([ds_word[story] for story in stories]), np.vstack([ds_WR[story] for story in stories])
    else: return [item for _, word in ds_word.items() if isinstance(word, Iterable) for item in word], [item for _, word in ds_WR.items() if isinstance(word, Iterable) for item in word]

def interp_word(stories, word_seq, config):
    """
    split all the words into sentence according to the fmri number
    """
    word_max_length = 0
    word_min_length = 10000
    ds_word = {}
    for story in stories:
        ds_story = []
        story_word = word_seq[story]
        ind_start = 0
        ind_end = 0
        for split_ind in story_word.split_inds:
            ind_end = split_ind
            if ind_end==ind_start:
                ds_story.append('' '')
                continue
            word_temp = ' '.join(story_word.data[ind_start: ind_end])
            word_max_length = max(word_max_length, ind_end-ind_start)
            word_min_length = min(word_min_length, ind_end-ind_start)
            ds_story.append(word_temp)
            ind_start = ind_end
        ds_word[story] = ds_story[4+config.trim:-config.trim]

    return ds_word, (word_min_length, word_max_length)
            

def predict_word_rate(resp, wt, vox, mean_rate):
    """predict word rate at each acquisition time
    """
    delresp = make_delayed(resp[:, vox], config.RESP_DELAYS)
    rate = ((delresp.dot(wt) + mean_rate)).reshape(-1).clip(min = 0)
    return np.round(rate).astype(int)

def predict_word_times(word_rate, resp, starttime = 0, tr = 2):
    """predict evenly spaced word times from word rate
    """
    half = tr / 2
    trf = TRFile(None, tr)
    trf.soundstarttime = starttime
    trf.simulate(resp.shape[0])
    tr_times = trf.get_reltriggertimes() + half

    word_times = []
    for mid, num in zip(tr_times, word_rate):  
        if num < 1: continue
        word_times.extend(np.linspace(mid - half, mid + half, num, endpoint = False) + half / num)
    return np.array(word_times), tr_times