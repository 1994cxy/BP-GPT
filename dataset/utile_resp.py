
import os
import numpy as np
import h5py
import copy
from typing import Iterable


def get_resp(sub, stories, config, stack = True, vox = None, only_info=True):
    subject_dir = os.path.join(config.data_dir, "preprocessed_data", sub)
    resp = {}
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        # print(resp_path)
        with h5py.File(resp_path, "r") as hf:
            temp_story = np.nan_to_num(hf["data"][:])
            if vox is not None:
                temp_story = temp_story[:, vox]
        win_num = len(temp_story)//config.hop_length
        win_num -= config.win_length//config.hop_length - 1  

        if not only_info:
            resp_story = np.zeros((win_num, config.win_length, temp_story.shape[-1]))
            for i in range(win_num):
                start = i*config.hop_length
                end = i*config.hop_length + config.win_length
                resp_story[i, :, :] = temp_story[start: end]
            resp[story] = resp_story

        else:
            story_win = []
            for i in range(win_num):
                start = i*config.hop_length
                end = i*config.hop_length + config.win_length
                story_win.append((resp_path, start, end))
            resp[story] = story_win
            
    if stack: return np.vstack([resp[story] for story in stories])
    else: return [item for _, word in resp.items() if isinstance(word, Iterable) for item in word]

