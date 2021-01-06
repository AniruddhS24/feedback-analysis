import torch
import torch.nn as nn
import math
from models.featurescorer import *

class Extractor(object):
    def __init__(self, featscorer):
        self.featscorer = featscorer

    def extract_rationales(self, x):
        raise Exception("Cannot instantiate Extractor base class. Please call one of "
                        + ", ".join([cls.__name__ for cls in Extractor.__subclasses__()]))

class HeuristicExtractor(Extractor):
    def __init__(self, featscorer, rationalelengthprop=0.1, num_rationales=3, rat_distance=0.05):
        super(HeuristicExtractor, self).__init__(featscorer)
        self.rationalelengthprop = rationalelengthprop
        self.num_rationales = num_rationales
        self.rat_distance = rat_distance

    def extract_rationales(self, x):
        scoreop = self.featscorer.get_score_features(x)
        bertops = scoreop["bert_outputs"]
        attnscores = scoreop["attention_scores"]
        inpids = scoreop["tokenizer"]["input_ids"]
        maxcontiglen = torch.sum(scoreop["tokenizer"]["attention_mask"],dim=1)

        rationale_data = {}
        rationales = []
        rationalevecs = []

        for batch_item in range(0,attnscores.shape[0]):
            stidxs = set()
            sentlen = maxcontiglen[batch_item].item()
            ratlen = round(sentlen * self.rationalelengthprop)
            for ridx in range(self.num_rationales):
                stidx, maxsum = 0, 0
                for i in range(1, sentlen-1-ratlen):
                    badspot = False
                    for j in stidxs:
                        if (i >= j and i < j+ratlen) or (i <= j and i+ratlen >= j):
                            badspot = True
                    if not badspot:
                        cursum = torch.sum(attnscores[batch_item, i:i+ratlen])
                        if cursum > maxsum:
                            stidx = i
                            maxsum = cursum

                # if feeding to BERT, will need in this format:
                # newinpids[batch_item, 0] = self.featscorer.tokenizer.cls_token_id
                # newinpids[batch_item, 1:ratlen+1] = inpids[batch_item, stidx:stidx+ratlen]
                # newinpids[batch_item, ratlen+1] = self.featscorer.tokenizer.sep_token_id
                # newinpids[batch_item, ratlen+2:] = self.featscorer.tokenizer.pad_token

                rel_dist = sentlen
                for stid in stidxs:
                    if stidx+ratlen <= stid:
                        rel_dist = min(rel_dist, stid-stidx+ratlen)
                    elif stid+ratlen <= stidx:
                        rel_dist = min(rel_dist, stidx- stid + ratlen)

                if rel_dist/sentlen >= self.rat_distance:
                    stidxs.add(stidx)
                    # otherwise we just want final BERT hidden states, so input ids not needed
                    rationalevecs.append((batch_item, torch.mean(bertops[batch_item, stidx:stidx+ratlen, :], dim=0), maxsum))
                    #rationales.append([self.featscorer.decode_inputids(inpids[batch_item, x]) for x in range(stidx, stidx+ratlen)])
                    rationales.append((batch_item, self.featscorer.decode_inputids(inpids[batch_item, stidx:stidx+ratlen]), maxsum))

        rationale_data["rationales"] = rationales
        rationale_data["rationale_avg_vec"] = rationalevecs
        # rationales: list of string rationales of form (batch_id, rationale, attn_score)
        # rationalevecs: list of averaged vectors of form (batch_id, averaged tensor, attn_score)
        return rationale_data

class BiLSTMNetwork(nn.Module):
    def __init__(self, inp_size, lstm_hid_size, num_layers=1, dropout=0.1):
        super(BiLSTMNetwork, self).__init__()
        if num_layers > 1:
            self.lstm = nn.LSTM(input_size=inp_size, hidden_size=lstm_hid_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout,bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=inp_size, hidden_size=lstm_hid_size, num_layers=num_layers,
                                batch_first=True, bidirectional=True)
        self.dense1 = nn.Linear(lstm_hid_size, 50)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(50, 2)
        self.act2 = nn.LogSoftmax()
        #self.act2 = nn.Softmax() change to this if needed

        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.kaiming_uniform_(self.dense2.weight)

    # bad loss function, don't use
    def mask_loss_soft(self, attns, ops, contiglens):
        # define loss here, should take into account contiguity, conciseness, and multiple rationales
        # not differentiable, need to minimize expected gradient by sampling binary masks
        lambda1 = 0.33
        lambda2 = 0.33
        lambda3 = 0.33

        #ops = ops.argmax(dim=2) to discretize predictions
        batchsz = ops.shape[0]
        seqlen = ops.shape[1]

        concise = torch.max(torch.zeros(batchsz),torch.div(torch.norm(ops[:,:,1], dim=1),torch.sqrt(contiglens)))

        contiguity = torch.zeros(batchsz)
        for i in range(1,seqlen):
            contiguity += torch.abs(torch.sub(ops[:,i,1], ops[:,i-1,1]))
        contiguity = torch.div(contiguity, contiglens+1)

        #attentionoverlap = -torch.log(torch.matmul(attns,torch.transpose(ops[:,:,1], 0, 1)))
        attentionoverlap = torch.zeros(batchsz)
        asrt = torch.argsort(attns,dim=1, descending=True)
        decay_rate = 0.1
        for t in range(seqlen):
            cbtch = torch.zeros(batchsz)
            for j in range(batchsz):
                cbtch[j] = -(1/math.exp(t*decay_rate))*math.log(ops[j,asrt[j][t],1])
            attentionoverlap += cbtch

        loss = lambda1*concise + lambda2*contiguity + lambda3*attentionoverlap
        return torch.mean(loss)

    # bad loss function, don't use
    def mask_loss_hard(self, attns, ops, contiglens):
        # define loss here, should take into account contiguity, conciseness, and multiple rationales
        # not differentiable, need to minimize expected gradient by sampling binary masks
        lambda1 = 0.33
        lambda2 = 0.33
        lambda3 = 0.33

        ops = ops.argmax(dim=2).float()
        batchsz = ops.shape[0]
        seqlen = ops.shape[1]

        concise = torch.max(torch.zeros(batchsz), torch.div(torch.norm(ops, dim=1), torch.sqrt(contiglens)))

        contiguity = torch.zeros(batchsz)
        for i in range(1, seqlen):
            contiguity += torch.abs(torch.sub(ops[:, i], ops[:, i - 1]))
        contiguity = torch.div(contiguity, contiglens + 1)

        attentionoverlap = torch.zeros(batchsz)
        asrt = torch.argsort(attns, dim=1, descending=True)
        decay_rate = 0.1
        for t in range(seqlen):
            cbtch = torch.zeros(batchsz)
            for j in range(batchsz):
                cbtch[j] = -(1/math.exp(t*decay_rate))*math.log(ops[j,asrt[j][t]])
            attentionoverlap += cbtch
        loss = lambda1 * concise + lambda2 * contiguity + lambda3 * attentionoverlap
        return torch.mean(loss)

    # best performance
    def pseudooutput_loss(self, labels, ops):
        obj = nn.NLLLoss()
        loss = 0.0
        for t in range(ops.shape[1]):
            loss += obj(ops[:, t], labels[:, t])
        return loss

    def forward(self, x):
        ops, (h_n,c_n) = self.lstm(x)
        ops = self.dense1(ops)
        ops = self.act1(ops)
        ops = self.dense2(ops)
        ops = self.act2(ops)
        return ops

class LSTMExtractor(Extractor):
    def __init__(self, featscorer, net:BiLSTMNetwork):
        super(LSTMExtractor, self).__init__(featscorer)
        self.net = net

if __name__ == '__main__':
    md = BiLSTMNetwork(10,20)
    ls = md.mask_loss_hard(attns=torch.tensor([[0.5,0.5]], requires_grad=True),
                      ops=torch.tensor([[[0.1,0.9],[0.1,0.9]]], requires_grad=True),
                      contiglens=torch.tensor([2.0,2.0], requires_grad=True))
    # lf = CustomLoss()
    # ls = lf.forward(ops=torch.tensor([1.0,2.0,3.0,4.0,5.0], requires_grad=True))
    ls.backward()