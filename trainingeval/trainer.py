from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import argparse
import yaml
from data.dataprocessing import *
from models.featurescorer import *
from models.extractor import *
from models.predictor import *
sys.path.pop(0)

def _parse_args():
    with open(r'trainingeval/config.yaml') as f:
        trainingconfig = yaml.full_load(f)

    parser = argparse.ArgumentParser(description='TRAINER')
    parser.add_argument('--model', type=str, help='model to train: (' + ', '.join(trainingconfig.keys()) + ')')
    parser.add_argument('--savepath', type=str, default='saved/nonamemodel.pt',
                        help='file path to save model state dict, set up as saved/[filename].pt')
    parser.add_argument('--dataset', type=str, default='YELP',
                        help='training dataset to use (YELP or TWITTER)')
    parser.add_argument('--num_samples', type=int, default=1000, help='number of training samples')
    parser.add_argument('--auxmodelsavepath', type=str, default=None, help='auxiliary model param file save path (ext or supp)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    modelname = args.model
    datasetname = args.dataset
    savefile = args.savepath
    num_samples = args.num_samples
    auxpath = args.auxmodelsavepath

    with open(r'config.yaml') as f:
        trainingconfig = yaml.full_load(f)

    if modelname == 'suppmodel':
        train_x, train_y, dev_x, dev_y = read_dataset(datasetname, num_samples)
        train_supp_model(train_x, train_y, dev_x, dev_y, FILENAME=savefile, device=device, config=trainingconfig[modelname])
    if modelname == 'lstmcrf':
        train_x, train_y, dev_x, dev_y = read_dataset(datasetname, num_samples)
        fsmodel = FeatureImportanceScorer(auxpath)
        train_lstmcrf_model(train_x, dev_x, fsmodel, FILENAME=savefile, device=device, config=trainingconfig[modelname])

    # sup = FeatureImportanceScorer('suppmodel2.pt')
    # scrop = sup.get_score_features(["I absolutely hate this place and it really sucks", "I love this place so much! It is amazing!"])
    # print(scrop['attention_scores'][0])
    # print(scrop['attention_scores'][1])
    # extmodel = HeuristicExtractor(sup)
    # rationales, ratvecs = extmodel.contiguous_discretize(["I absolutely hate this place and it really sucks", "I love this place so much! It is amazing!"])
    # print(rationales)