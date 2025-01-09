
import numpy as np
from bert_score import BERTScorer
from jiwer import wer

"""
WER
"""
class WER(object):
    def __init__(self, use_score = True):
        self.use_score = use_score
    
    def score(self, ref, pred):
        scores = []
        for ref_seg, pred_seg in zip(ref, pred):
            if len(ref_seg) == 0 : error = 1.0
            else: error = wer(ref_seg, pred_seg)
            if self.use_score: scores.append(1 - error)
            else: scores.append(error)
        return np.array(scores)
    
"""
BERTScore (https://arxiv.org/abs/1904.09675)
"""
class BERTSCORE(object):
    def __init__(self, idf_sents=None, rescale = True, score = "f"):
        self.metric = BERTScorer(lang = "en", rescale_with_baseline = rescale, idf = (idf_sents is not None), idf_sents = idf_sents)
        if score == "precision": self.score_id = 0
        elif score == "recall": self.score_id = 1
        else: self.score_id = 2

    def score(self, ref, pred):
        # ref_strings = [" ".join(x) for x in ref]
        # pred_strings = [" ".join(x) for x in pred]
        return self.metric.score(cands = pred, refs = ref)[self.score_id].numpy()
    

"""windows of [duration] seconds at each time point"""
def windows(start_time, end_time, duration, step = 1):
    start_time, end_time = int(start_time), int(end_time)
    half = int(duration / 2)
    return [(center - half, center + half) for center in range(start_time + half, end_time - half + 1) if center % step == 0]

"""divide [data] into list of segments defined by [cutoffs]"""
def segment_data(data, times, cutoffs):
    return [[x for c, x in zip(times, data) if c >= start and c < end] for start, end in cutoffs]

if __name__=='__main__':
    pred = ['on the highway and i']
    ref = ['the door and i ran']
    bert_score = BERTSCORE(idf_sents = None, rescale = True, score = "recall")
    bert_score.score(ref, pred)

