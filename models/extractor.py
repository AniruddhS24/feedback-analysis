import torch
import torch.nn as nn
from models.featurescorer import *

class HeuristicExtractor:
    def __init__(self, featscorer:FeatureImportanceScorer):
        self.featscorer = featscorer

    def contiguous_discretize(self, x, rationalelengthprop=0.20):
        scoreop = self.featscorer.get_score_features(x)
        bertops = scoreop["bert_outputs"]
        attnscores = scoreop["attention_scores"]
        inpids = scoreop["tokenizer"]["input_ids"]
        maxcontiglen = torch.sum(scoreop["tokenizer"]["attention_mask"],dim=1)

        rationales = []
        rationalevecs = torch.zeros((bertops.shape[0], bertops.shape[2]))

        for batch_item in range(0,attnscores.shape[0]):
            stidx, maxsum = 0, 0
            sentlen = maxcontiglen[batch_item].item()
            ratlen = round(sentlen*rationalelengthprop)
            for i in range(1, sentlen-1-ratlen):
                cursum = torch.sum(attnscores[batch_item, i:i+ratlen])
                if cursum > maxsum:
                    stidx = i
                    maxsum = cursum

            # if feeding to BERT, will need in this format:
            # newinpids[batch_item, 0] = self.featscorer.tokenizer.cls_token_id
            # newinpids[batch_item, 1:ratlen+1] = inpids[batch_item, stidx:stidx+ratlen]
            # newinpids[batch_item, ratlen+1] = self.featscorer.tokenizer.sep_token_id
            # newinpids[batch_item, ratlen+2:] = self.featscorer.tokenizer.pad_token

            # otherwise we just want final BERT hidden states, so input ids not needed
            rationalevecs[batch_item] = torch.mean(bertops[batch_item, stidx:stidx+ratlen, :], dim=0)
            #rationales.append([self.featscorer.decode_inputids(inpids[batch_item, x]) for x in range(stidx, stidx+ratlen)])
            rationales.append(self.featscorer.decode_inputids(inpids[batch_item, stidx:stidx+ratlen]))

        # rationales: list of string rationales
        # rationalevecs: (batchsize, hiddensize) averaged bert embedding vectors
        return rationales, rationalevecs