import sys
from data.dataprocessing import *
from models.featurescorer import *
from models.extractor import *
from models.predictor import *

def train_supp(savefile, num_samples, device):
    train_x, train_y, dev_x, dev_y = read_dataset('YELP', num_samples)
    train_supp_model(train_x, train_y, dev_x, dev_y, FILENAME=savefile, device=device)
    #sup = FeatureImportanceScorer('suppmodel.pt')

def train_pred():
    # supmodel = FeatureImportanceScorer('suppmodel.pt')
    # extmodel = HeuristicExtractor(supmodel)
    # predmodel = SentimentPredictor(extmodel)
    pass

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    savefile = sys.argv[1]
    num_samples = int(sys.argv[2])
    train_supp(savefile, num_samples, device)
    # sup = FeatureImportanceScorer('suppmodel2.pt')
    # scrop = sup.get_score_features(["I absolutely hate this place and it really sucks", "I love this place so much! It is amazing!"])
    # print(scrop['attention_scores'][0])
    # print(scrop['attention_scores'][1])
    # extmodel = HeuristicExtractor(sup)
    # rationales, ratvecs = extmodel.contiguous_discretize(["I absolutely hate this place and it really sucks", "I love this place so much! It is amazing!"])
    # print(rationales)