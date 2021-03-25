import sys
import pickle
from models.featurescorer import *
from models.extractor import *
from models.predictor import *
from flask import Flask, request, jsonify
# from waitress import serve
# from flask_cors import CORS

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
            compact_op.append((preds[i].item(), rats[i][1]))
        return compact_op

def save_feedbackmodel(save_path):
    mainmodel = FeedbackModel('saved/suppmodeelnew.pt',
                              'saved/heuristicext.pt',
                              'saved/testdn.pt')
    with open(save_path, 'wb') as f:
        pickle.dump(mainmodel, f)

'''curl -i -H "Content-Type: application/json" -X POST -d '{"input": "The movie was ok"}' http://127.0.0.1:5000/'''
@app.route('/', methods=['POST'])
def run_inference():
    data = request.json
    output_list = fbmodel.predict(data['input'])
    output = {}
    output['output_rationales'] = []
    output['output_logits'] = []
    for x in output_list:
        output['output_rationales'].append(x[1])
        output['output_logits'].append(x[0])
    return jsonify(output)

def deploy_backend_dev():
    with open('feedbackmodel.pickle', 'rb') as f:
        fbmodel = pickle.load(f)
    app.run(host='0.0.0.0', port=80)

'''
must import relevant packages first (waitress and flask_cors), not in requirements.txt
def deploy_backend_production():
    CORS(app)
    with open('feedbackmodel.pickle', 'rb') as f:
        fbmodel = pickle.load(f)
    serve(app, host='0.0.0.0', port=48932, url_scheme='https')
'''

def deploy_model():
    inputstr = sys.argv[1]
    # TODO: DAN with bert averages doesn't work well... maybe train BERT pred or use glove embeddings instead
    with open('myfeedbackmodel.pickle', 'rb') as f:
        unserialized_data = pickle.load(f)

    mydata = unserialized_data.predict(inputstr)
    for x in mydata:
        print(str(x[0]) + "\t" + x[1])

if __name__ == '__main__':
    deploy_model()