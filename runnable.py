import sys
import pickle
from models.featurescorer import *
from models.extractor import *
from models.predictor import *
from flask import Flask, request, jsonify

app = Flask(__name__)
fbmodel = None

class FeedbackModel:
    def __init__(self, supp_model_path, ext_model_path, pred_model_path):
        self.supp = load_featurescorer_model(model_file_path=supp_model_path)
        self.ext = load_extractor_model(model_file_path=ext_model_path,
                                        featscorer=self.supp)
        self.pred = load_predictor_model(model_file_path=pred_model_path,
                                         extractor=self.ext)

    def predict(self, x):
        compact_op = []
        preds, rats = self.pred.predict([x])
        for i in range(len(preds)):
            compact_op.append((preds[i], rats[i][1]))
        return compact_op

def save_feedbackmodel(save_path):
    mainmodel = FeedbackModel('saved/suppmodeelnew.pt',
                              'saved/heuristicext.pt',
                              'saved/testdn.pt')
    with open(save_path, 'wb') as f:
        pickle.dump(mainmodel, f, protocol=pickle.HIGHEST_PROTOCOL)

'''curl -i -H "Content-Type: application/json" -X POST -d '{"input": "The movie was ok"}' http://127.0.0.1:5000/'''
@app.route('/', methods=['POST'])
def run_inference():
    data = request.json
    output_list = fbmodel.predict(data['input'])
    output = {}
    output['output_rationales'] = []
    for x in output_list:
        output['output_rationales'].append(x[1])
    return jsonify(output)

if __name__ == '__main__':
    with open('myfeedbackmodel.pickle', 'rb') as f:
        fbmodel = pickle.load(f)
    app.run(debug=True)


'''
if __name__ == '__main__':
    inputstr = sys.argv[1]
    # TODO: DAN with bert averages doesn't work well... maybe train BERT pred or use glove embeddings instead
    with open('myfeedbackmodel.pickle', 'rb') as f:
        unserialized_data = pickle.load(f)

    mydata = unserialized_data.predict(inputstr)
    for x in mydata:
        print(str(x[0]) + "\t" + x[1])
'''