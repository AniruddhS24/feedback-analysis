import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from models.extractor import *

'''
Really this is a DAN which averages BERT embedding outputs... may not be a good choice
Consider making another BERT module where only rationales are passed and binary classification fine-tuning
The pooled output can then be used
'''
class DAN(nn.Module):
    def __init__(self, inp_size, hid_size, op_size):
        super(DAN, self).__init__()

        # note: inp_size is the bert hidden dimension (output from extractor)
        self.dense1 = nn.Linear(inp_size, hid_size)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(hid_size, op_size)
        self.act2 = nn.LogSoftmax()

        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.kaiming_uniform_(self.dense2.weight)

    def forward(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return x

class BERTPred(nn.Module):
    def __init__(self, bert_model_type, freeze=False):
        super(BERTPred, self).__init__()

        self.model_name = bert_model_type
        self.bert_config = AutoConfig.from_pretrained(bert_model_type,
                                                      output_attentions=True)  # TODO: change to BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
        self.bert = AutoModel.from_pretrained(bert_model_type, config=self.bert_config)
        self.dpout = nn.Dropout(p=0.1)  # regularisation
        self.dense1 = nn.Linear(self.bert.config.hidden_size, 50)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(50, 2)
        self.softmx = nn.LogSoftmax(dim=1)

        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.kaiming_uniform_(self.dense2.weight)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def convert_for_bert_input(self, rationales):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # fullrecon = []
        # for batch_item in rationales:
        #     reconstructed = " ".join(rationales[batch_item]).replace(" ##","")
        #     fullrecon.append(reconstructed)
        return tokenizer(text=rationales,
                  add_special_tokens=True,
                  padding='max_length',
                  return_tensors='pt')["input_ids"]

    # input is of shape (batchsize, seqlen) - must be tokenized beforehand
    def forward(self, inpids):
        outputs = self.bert(input_ids=inpids)
        cumout = outputs['pooler_output']
        x = self.dpout(cumout)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        op = self.softmx(x)
        # output is of shape (batchsize, 2)
        return op

class SentimentPredictor:
    # TODO: make the extractor class more general (extractor superclass), should work with all extractor models
    '''
    Takes as input a TRAINED predictor network and an extractor model
    '''
    def __init__(self, net, extractor:HeuristicExtractor):
        self.net = net
        self.extractor = extractor

    def predict(self, x):
        rationales, rationalevecs = self.extractor.contiguous_discretize(x, rationalelengthprop=0.2)
        if (isinstance(self.net, BERTPred)):
            pred = self.net.forward(self.net.convert_for_bert_input(rationales)).argmax(dim=1)
        elif (isinstance(self.net, DAN)):
            pred = self.net.forward(rationalevecs).argmax(dim=1) # 1/0 predictions
        else:
            pred = None
        return rationales, pred