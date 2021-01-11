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
    #fbmd = FeedbackModel()
    #pickle.dump(fbmd, open('mymodel.pickle', 'wb'))
    item = pickle.load(open('mymodel.pickle', 'rb'))
    print(item.predict("In many settings it is important for one to be able to understand why a model made a particular prediction. In NLP this often entails extracting snippets of an input text responsible"))