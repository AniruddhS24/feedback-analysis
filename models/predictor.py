import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from models.extractor import *

class Predictor(object):
    def __init__(self, extractor):
        self.extractor = extractor

    def predict(self, x, soft=False):
        raise Exception("Cannot instantiate Extractor base class. Please call one of "
                        + ", ".join([cls.__name__ for cls in Predictor.__subclasses__()]))

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
        self.act2 = nn.LogSoftmax(dim=1)

        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.kaiming_uniform_(self.dense2.weight)

    def forward(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return x

class DANPredictor(Predictor):
    def __init__(self, extractor, checkpoint):
        super(DANPredictor, self).__init__(extractor)
        self.net = self.load_model(checkpoint)

    def load_model(self, checkpt):
        try:
            net = DAN(inp_size=checkpt["config"]["inp_size"],
                           hid_size=checkpt["config"]["hid_size"],
                           op_size=checkpt["config"]["op_size"])
            net.load_state_dict(checkpt['state_dict'])
        except:
            raise Exception("Error! Model checkpoint file not found, or file not saved correctly.")
        return net

    def predict(self, x, soft=False):
        _, rationale_data = self.extractor.extract_rationales(x)
        xbatch = torch.zeros(len(rationale_data["rationale_avg_vec"]),
                             rationale_data["rationale_avg_vec"][0][1].shape[0])
        for j in range(len(rationale_data["rationale_avg_vec"])):
            xbatch[j] = rationale_data["rationale_avg_vec"][j][1]

        pred = self.net(xbatch)

        if soft:
            return pred, rationale_data["rationales"]
        else:
            return pred.argmax(dim=1), rationale_data["rationales"]

class BERTPred(nn.Module):
    def __init__(self, bert_model_type, freeze=False):
        super(BERTPred, self).__init__()

        self.model_name = bert_model_type
        self.bert_config = AutoConfig.from_pretrained(bert_model_type,
                                                      output_attentions=True)
        self.bert = AutoModel.from_pretrained(bert_model_type, config=self.bert_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dpout = nn.Dropout(p=0.1)  # regularisation
        self.dense1 = nn.Linear(self.bert_config.hidden_size, 50)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(50, 2)
        self.softmx = nn.LogSoftmax(dim=1)
    
        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.kaiming_uniform_(self.dense2.weight)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def process_input(self, textrats):
        # textrats = [rationales[i][1] for i in range(len(rationales))]
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # fullrecon = []
        # for batch_item in rationales:
        #     reconstructed = " ".join(rationales[batch_item]).replace(" ##","")
        #     fullrecon.append(reconstructed)
        return self.tokenizer.encode(text=textrats,
                     add_special_tokens=True,
                     padding='max_length',
                     truncation='only_first',
                     return_tensors='pt')['input_ids']

    # input is of shape (batchsize, seqlen) - must be tokenized beforehand
    def forward(self, inpids):
        outputs = self.bert(input_ids=inpids)
        cumout = outputs[1] # pooler_output
        x = self.dpout(cumout)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        op = self.softmx(x)
        # output is of shape (batchsize, 2)
        return op

class BERTPredictor(Predictor):
    def __init__(self, extractor, checkpoint):
        super(BERTPredictor, self).__init__(extractor)
        self.net = self.load_model(checkpoint)

    def load_model(self, checkpt):
        try:
            net = BERTPred(bert_model_type=checkpt["config"]["bert_model_type"],
                           freeze=checkpt["config"]["freeze"])
            net.load_state_dict(checkpt['state_dict'])
        except:
            raise Exception("Error! Model checkpoint file not found, or file not saved correctly.")
        return net

    def predict(self, x, soft=False):
        _, rationale_data = self.extractor.extract_rationales(x)
        pred = self.net.forward(self.net.process_input(rationale_data["rationales"]))
        if soft:
            return pred, rationale_data["rationales"]
        else:
            return pred.argmax(dim=1), rationale_data["rationales"]

def evaluate_dan_pred_model(net, extmodel, dev_x, dev_y, device):
    print("Evaluating DAN Pred Model...")
    net.eval()
    # dev set accuracy
    batch_size = min(10, len(dev_x)-1)
    num_correct = 0
    num_total = 0
    for i in range(batch_size, len(dev_x), batch_size):
        xbatch_str = dev_x[i - batch_size:i]
        _, rationale_data = extmodel.extract_rationales(xbatch_str)

        xbatch = torch.zeros(len(rationale_data["rationale_avg_vec"]),
                             rationale_data["rationale_avg_vec"][0][1].shape[0])
        ybatch = torch.zeros(len(rationale_data["rationale_avg_vec"]))
        for j in range(len(rationale_data["rationale_avg_vec"])):
            xbatch[j] = rationale_data["rationale_avg_vec"][j][1]
            ybatch[j] = dev_y[i - batch_size + rationale_data["rationale_avg_vec"][j][0]]
        xbatch, ybatch = xbatch.to(device), ybatch.long().to(device)

        pred_lbl = net(xbatch).argmax(dim=1)
        for j in range(len(pred_lbl)):
            num_total += 1
            if pred_lbl[j]==ybatch[j]:
                num_correct += 1
    acc = num_correct / num_total
    print("Dev Accuracy: {0} ({1} / {2})".format(acc * 100, num_correct, num_total))
    return acc

# expects train_x as list of strings, train_y as one-hot encoded label tensor
def train_dan_pred_model(train_x, train_y, dev_x, dev_y, extmodel, FILENAME, device, config):
    net = DAN(config['inp_size'], config['hid_size'], config['op_size'])
    net = net.to(device)

    # hyperparameters
    EPOCHS = config['EPOCHS']
    batch_size = config['batch_size']
    lr = config['lr']
    objective = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=config['l2reg'])

    print("Training DAN Pred Model...")

    # training
    for epoch in range(EPOCHS):
        tot_loss = 0.0
        perm = torch.randperm(len(train_x))
        net.train()
        for i in range(batch_size, len(perm), batch_size):
            optimizer.zero_grad()
            xbatch_str = [train_x[j] for j in perm[i - batch_size:i]]
            _, rationale_data = extmodel.extract_rationales(xbatch_str)
            xbatch = torch.zeros(len(rationale_data["rationale_avg_vec"]),
                                 rationale_data["rationale_avg_vec"][0][1].shape[0])
            ybatch = torch.zeros(len(rationale_data["rationale_avg_vec"]))
            for j in range(len(rationale_data["rationale_avg_vec"])):
                xbatch[j] = rationale_data["rationale_avg_vec"][j][1]
                ybatch[j] = train_y[perm[i - batch_size + rationale_data["rationale_avg_vec"][j][0]]]
            xbatch, ybatch = xbatch.to(device), ybatch.long().to(device)

            pred = net(xbatch)
            loss = objective(pred, ybatch)
            tot_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            del loss, pred, rationale_data, xbatch, ybatch, xbatch_str
        print("Epoch {0}      Loss {1}".format(epoch, tot_loss))
        if epoch%5==0:
            acc = evaluate_dan_pred_model(net, extmodel, dev_x,dev_y, device)
            checkpt = {
                'state_dict': net.state_dict(),
                'accuracy': acc*100,
                'config': config}
            torch.save(checkpt, FILENAME[0:FILENAME.index('.')] + str(epoch) + 'epoch.pt')

    acc = evaluate_dan_pred_model(net, extmodel, dev_x, dev_y, device)
    checkpt = {
        'state_dict': net.state_dict(),
        'accuracy': acc*100,
        'config': config}
    torch.save(checkpt, FILENAME)

def evaluate_bert_pred_model(net, extmodel, dev_x, dev_y, device):
    print("Evaluating BERT Pred Model...")
    net.eval()
    # dev set accuracy
    batch_size = min(10, len(dev_x)-1)
    num_correct = 0
    num_total = 0
    for i in range(batch_size, len(dev_x), batch_size):
        xbatch_str = dev_x[i - batch_size:i]
        xbatch = net.process_input(xbatch_str)
        ybatch = dev_y[i - batch_size:i]
        xbatch, ybatch = xbatch.to(device), ybatch.long().to(device)

        pred_lbl = net(xbatch).argmax(dim=1)
        for j in range(len(pred_lbl)):
            num_total += 1
            if pred_lbl[j]==ybatch[j]:
                num_correct += 1
    acc = num_correct / num_total
    print("Dev Accuracy: {0} ({1} / {2})".format(acc * 100, num_correct, num_total))
    return acc

def train_bert_pred_model(train_x, train_y, dev_x, dev_y, FILENAME, device, config):
    net = BERTPred(bert_model_type = config['bert_model_name'])
    net = net.to(device)

    # hyperparameters
    EPOCHS = config['EPOCHS']
    batch_size = config['batch_size']
    lr = config['lr']
    objective = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=config['l2reg'])

    print("Training BERT Pred Model...")
    # training
    for epoch in range(EPOCHS):
        tot_loss = 0.0
        perm = torch.randperm(len(train_x))
        net.train()
        for i in range(0, len(perm), batch_size):
            optimizer.zero_grad()
            xbatch_str = [train_x[j] for j in perm[i - batch_size:i]]
            xbatch = net.process_input(xbatch_str)
            ybatch = train_y[perm[i - batch_size:i]]
            xbatch, ybatch = xbatch.to(device), ybatch.long().to(device)

            pred = net(xbatch)
            loss = objective(pred, ybatch)
            tot_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            del loss, pred, xbatch, ybatch, xbatch_str

        print("Epoch {0}      Loss {1}".format(epoch, tot_loss))
        if epoch%5==0:
            acc = evaluate_bert_pred_model(net, dev_x, dev_y, device)
            checkpt = {
                'state_dict': net.state_dict(),
                'accuracy': acc*100,
                'config': config}
            torch.save(checkpt, FILENAME[0:FILENAME.index('.')] + str(epoch) + 'epoch.pt')

    acc = evaluate_bert_pred_model(net, dev_x, dev_y, device)
    checkpt = {
        'state_dict': net.state_dict(),
        'accuracy': acc*100,
        'config': config}
    torch.save(checkpt, FILENAME)


def load_predictor_model(model_file_path, extractor):
    checkpoint = torch.load(model_file_path)
    model_name = checkpoint['config']['model_name']

    if model_name == 'danpred':
        return DANPredictor(extractor=extractor, checkpoint=checkpoint)
    if model_name == 'bertpred':
        return BERTPredictor(extractor=extractor, checkpoint=checkpoint)

    del checkpoint
    raise Exception("Error loading model")