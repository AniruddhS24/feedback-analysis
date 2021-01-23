import sys
import pickle
from models.featurescorer import *
from models.extractor import *
from models.predictor import *

class FeedbackModel:
    def __init__(self, supp_model_path, ext_model_path, pred_model_path):
        self.supp = load_featurescorer_model(model_file_path=supp_model_path)
        self.ext = load_extractor_model(model_file_path=ext_model_path,
                                        featscorer=self.supp)
        #self.pred = load_predictor_model(model_file_path=pred_model_path,
                                         #extractor=self.ext)

    def predict(self, x):
        _, rationale_data = self.ext.extract_rationales([x])
        return rationale_data["rationales"]

if __name__ == '__main__':
    #inputstr = sys.argv[1]
    mainmodel = FeedbackModel('saved/suppmodeelnew.pt',
                              'saved/heuristicext.pt',
                              'saved/danpred.pt')
    with open('testmodel.pickle', 'wb') as f:
        pickle.dump(mainmodel,f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('testmodel.pickle', 'rb') as f:
    #     unserialized_data = pickle.load(f)