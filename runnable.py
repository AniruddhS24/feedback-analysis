import sys
import pickle
import torch
import torch.nn as nn
from models.featurescorer import *
from models.extractor import *
from transformers import AutoModel, AutoConfig, AutoTokenizer

class FeedbackModel:
    def __init__(self):
        self.ext = HeuristicExtractor(featscorer=FeatureImportanceScorer('saved/suppmodeel.pt'))

    def predict(self, x):
        _, rationale_data = self.ext.extract_rationales([x])
        return rationale_data["rationales"]

if __name__ == '__main__':
    inputstr = sys.argv[1]
    fs = load_featurescorer_model('saved/suppmodeelnew.pt')
    model = load_extractor_model('saved/lstmcrfmodel.pt', fs)
    print(model.extract_rationales([inputstr])["rationales"])