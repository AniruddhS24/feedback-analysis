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
    parser.add_argument('--auxfeatscorer', type=str, default=None, help='auxiliary model param file save path')
    parser.add_argument('--auxextractor', type=str, default=None, help='auxiliary model param file save path')
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
    auxfs = args.auxfeatscorer
    auxext = args.auxextractor

    with open(r'config.yaml') as f:
        trainingconfig = yaml.full_load(f)

    train_x, train_y, dev_x, dev_y = read_dataset(datasetname, num_samples)

    if modelname == 'suppmodel':
        train_supp_model(train_x, train_y, dev_x, dev_y, FILENAME=savefile, device=device, config=trainingconfig[modelname])
    if modelname == 'heuristic':
        train_heuristic_model(FILENAME=savefile, config=trainingconfig[modelname])
    if modelname == 'lstmcrf':
        fsmodel = load_featurescorer_model(auxfs) #TODO: edit this checkpoint file to contain new fields, retraining too expensive
        train_lstmcrf_model(train_x, dev_x, fsmodel, FILENAME=savefile, device=device, config=trainingconfig[modelname])
    if modelname == 'danpred':
        fsmodel = load_featurescorer_model(auxfs)
        extmodel = load_extractor_model(fsmodel, auxext)
        train_dan_pred_model(train_x, train_y, dev_x, dev_y, extmodel, FILENAME=savefile, device=device, config=trainingconfig[modelname])
