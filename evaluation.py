import sys
from data.dataprocessing import *
from models.featurescorer import *
from models.extractor import *
from models.predictor import *

def evaluate_supp(savefile, num_samples, device):
    train_x, train_y, dev_x, dev_y = read_dataset('YELP', num_samples)
    model = FeatureImportanceScorer(model_param_path=savefile)
    evaluate_supp_model(model, dev_x, dev_y, device)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    savefile = sys.argv[1]
    num_samples = int(sys.argv[2])
    evaluate_supp(savefile, num_samples, device)