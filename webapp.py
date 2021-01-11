import os
import pickle
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from runnable import *
from flask import Flask
from flask import request
from flask import json

os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

BUCKET_NAME = 'fbanalysisbucket'
MODEL_FILE_NAME = 'newmodel.pickle'
MODEL_LOCAL_PATH = 'temp.pickle'

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
  payload = json.loads(request.get_data().decode('utf-8'))
  prediction = predict(payload['payload'])
  data = {}
  data['data'] = prediction
  return json.dumps(data)

def load_model():
  conn = S3Connection(aws_access_key_id='AKIAJUOGPBCGN34USBRQ', aws_secret_access_key='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', host='s3.us-east-2.amazonaws.com')
  #print("connection made")
  bucket = conn.get_bucket(BUCKET_NAME)
  key_obj = Key(bucket)
  key_obj.key = MODEL_FILE_NAME
  #print("bucket stuff done")
  contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
  #print("downloaded contents")
  return pickle.load(open(MODEL_LOCAL_PATH, 'rb'))

def predict(data):
  mdl = load_model()
  #print("model loaded")
  preds = mdl.predict(data)
  final_formatted_data = {}
  final_formatted_data["rationales"] = [x[1] for x in preds]
  final_formatted_data["scores"] = [x[2] for x in preds]
  return final_formatted_data

if __name__ == '__main__':
  fbmd = FeedbackModel()
  pickle.dump(fbmd, open('newmodel.pickle', 'wb'))