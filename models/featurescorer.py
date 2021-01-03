import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

class SuppModel(nn.Module):
    def __init__(self, bert_model_type, freeze=False):
        super(SuppModel, self).__init__()
        self.model_name = bert_model_type
        self.bert_config = AutoConfig.from_pretrained(bert_model_type,
                                                      output_attentions=True)  # TODO: change to BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
        self.bert = AutoModel.from_pretrained(bert_model_type, config=self.bert_config)
        self.dpout = nn.Dropout(p=0.1) # regularisation
        self.dense1 = nn.Linear(self.bert.config.hidden_size, 50)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(50, 2)
        self.softmx = nn.LogSoftmax(dim=1)

        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.kaiming_uniform_(self.dense2.weight)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def get_outputs_and_attentions(self, inpids):
        output = self.bert(input_ids=inpids)
        # out = output['last_hidden_state']
        # attns = output['attentions']
        out = output[0]
        attns = output[2]
        # attns of shape (layers, batchsize, attnheads, seqlen, seqlen)
        res = attns[-1][:, :, 0, :]
        # take average along attention heads
        # return tensor of shape (batchsize, seqlen)
        return out, torch.mean(res,dim=1)


    # input is of shape (batchsize, seqlen) - must be tokenized beforehand
    def forward(self, inpids):
        output = self.bert(input_ids=inpids)
        #cumout = output['pooler_output']
        cumout = output[1]
        x = self.dpout(cumout)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        op = self.softmx(x)
        # output is of shape (batchsize, 2)
        return op

class FeatureImportanceScorer:
    '''
    Takes as input a trained SuppModel
    '''
    def __init__(self, model_param_path=None):
        self.net = self.load_model(model_param_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.net.model_name)

    def load_model(self, filename):
        net = SuppModel(bert_model_type='bert-base-uncased', freeze=False)
        if filename is None:
            return net
        checkpt = torch.load("saved/" + filename)
        net.load_state_dict(checkpt['state_dict'])
        return net

    def process_input(self, x):
        # input is list of strings (batchsize, strings)
        return self.tokenizer(text=x,
                     add_special_tokens=True,
                     padding='max_length',
                     truncation='only_first',
                     return_attention_mask=True,
                     return_tensors='pt')

    '''
    Expects inpids as 1-dimensional List[int]
    '''
    def decode_inputids(self, inpids):
        return self.tokenizer.decode(inpids)

    def get_score_features(self, x):
        self.net.eval()
        tknx = self.process_input(x) # tknx: (batchsize, seqlen)
        ops,attns = self.net.get_outputs_and_attentions(tknx["input_ids"]) # attns: (batchsize, seqlen)
        scoreop = dict()
        scoreop["tokenizer"] = tknx
        scoreop["bert_outputs"] = ops
        scoreop["attention_scores"] = attns
        return scoreop

def train_supp_model(train_x, train_y, dev_x, dev_y, FILENAME, device):
    model = FeatureImportanceScorer()
    net = model.net
    #net = net.to(device)
    train_x = model.process_input(train_x)["input_ids"]
    train_y = torch.tensor(train_y)

    # hyperparameters
    EPOCHS = 20
    batch_size = 32
    lr = 2e-5
    objective = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)

    print("Training Supp Model...")
    net.train()
    # training
    for epoch in range(EPOCHS):
        tot_loss = 0.0
        perm = torch.randperm(train_x.shape[0])
        for i in range(batch_size, len(perm), batch_size):
            optimizer.zero_grad()
            xbatch = train_x[perm[i-batch_size:i]]
            ybatch = train_y[perm[i-batch_size:i]]
            #xbatch, ybatch = xbatch.to(device), ybatch.to(device)
            pred = net.forward(xbatch)
            loss = objective(pred, ybatch)
            tot_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            del loss, pred
        print("Epoch {0}      Loss {1}".format(epoch, tot_loss))
        if epoch%4==0:
            checkpt = {
                'state_dict': net.state_dict(),
                'accuracy': 0,
                'epochs': EPOCHS,
                'batch_size': batch_size,
                'lr': lr}
            torch.save(checkpt, FILENAME[0:FILENAME.index('.')] + str(epoch) + 'epoch.pt')


    # path = F"/content/gdrive/My Drive/suppmodel.pt"
    # torch.save(net.state_dict(), path)

    net.eval()
    # dev set accuracy
    dev_x = model.process_input(dev_x)["input_ids"]
    dev_y = torch.tensor(dev_y)
    num_cor = 0
    for i in range(dev_x.shape[0]):
        pred_lbl = net.forward(dev_x[i].unsqueeze(dim=0)).squeeze().argmax(dim=0)
        if pred_lbl == dev_y[i]:
            num_cor+=1
    acc = num_cor/dev_x.shape[0]
    print("Dev Accuracy: {0}".format(acc*100))

    checkpt = {
        'state_dict': net.state_dict(),
        'accuracy': acc*100,
        'epochs': EPOCHS,
        'batch_size': batch_size,
        'lr': lr}
    torch.save(checkpt, FILENAME)


