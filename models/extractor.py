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

'''
Extractor solely based on heuristics, no model
'''
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
        binarymask = torch.zeros(bertops.shape[0], bertops.shape[1])

        for batch_item in range(0,bertops.shape[0]):
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
                    binarymask[batch_item, stidx:stidx+ratlen] = 1

        rationale_data["rationales"] = rationales
        rationale_data["rationale_avg_vec"] = rationalevecs
        rationale_data["binary_mask"] = binarymask
        # rationales: list of string rationales of form (batch_id, rationale, attn_score)
        # rationalevecs: list of averaged vectors of form (batch_id, averaged tensor, attn_score)
        return rationale_data

'''
Parameterized bare-LSTM tagging model
'''
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

    # supervised on heuristics from feature scoring model
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

'''
Parameterized LSTM with a CRF layer
https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea
'''
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # TODO: see if pad token transitions need to be manually taken care of

    def forward(self, emissions, tags, mask):
        z = self.compute_partition(emissions, mask=mask)
        scores = self.compute_scores(emissions, tags, mask=mask)
        nll = -torch.mean(z - scores)
        return nll

    def compute_scores(self, emissions, tags, mask=None):
        batch_size, seqlen, num_tags = emissions.shape
        score = torch.zeros(batch_size)
        for t in range(1,seqlen):
            mask_t = mask[:, t]
            emit_t = torch.tensor([ems[t, tgs[t]] for ems, tgs in zip(emissions, tags)])
            trans_t = torch.tensor([self.transitions[tgs[t - 1], tgs[t]] for tgs in tags])
            score += (emit_t + trans_t) * mask_t
        return score # return score (batchsz,) tensor

    def compute_partition(self, emissions, mask=None):
        batch_size, seq_length, nb_labels = emissions.shape
        dp = torch.zeros(batch_size, nb_labels)

        for i in range(seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = dp.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            dp_t = torch.logsumexp(scores, dim=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            dp = is_valid * dp_t + (1 - is_valid) * dp

        return torch.logsumexp(dp, dim=1)

    def decode(self, emissions, mask=None):
        batch_size, seq_length, nb_labels = emissions.shape
        dp = torch.zeros(batch_size, nb_labels)

        backpointers = []

        for i in range(seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)
            # (nb_labels, nb_labels) -> (1, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)
            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = dp.unsqueeze(2)
            # combine current scores with previous dp
            scores = e_scores + t_scores + a_scores

            # so far is exactly like the forward algorithm,
            # but now, instead of calculating the logsumexp,
            # we will find the highest score and the tag associated with it
            max_scores, max_score_tags = torch.max(scores, dim=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            dp = is_valid * max_scores + (1 - is_valid) * dp

            # add the max_score_tags for our list of backpointers
            # max_scores has shape (batch_size, nb_labels) so we transpose it to
            # be compatible with our previous loopy version of viterbi
            backpointers.append(max_score_tags.t())


        max_final_scores, max_final_tags = torch.max(dp, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self.backtrack_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def backtrack_path(self, sample_id, best_tag, backpointers):
        # add the final best_tag to our best path
        best_path = [best_tag]
        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):
            best_tag = backpointers_t[best_tag][sample_id].item()
            best_path.insert(0, best_tag)
        return best_path

class LSTM_CRF(nn.Module):
    def __init__(self, inp_size, lstm_hid_size, num_layers=1, dropout=0.1):
        super(LSTM_CRF, self).__init__()
        self.rnn = BiLSTMNetwork(inp_size=inp_size, lstm_hid_size=lstm_hid_size, num_layers=num_layers,
                            dropout=dropout)
        self.crf = CRF(num_tags=2)

    # supervised on heuristics from feature scoring model
    def pseudooutput_loss(self, labels, ops):
        obj = nn.NLLLoss()
        loss = 0.0
        for t in range(ops.shape[1]):
            loss += obj(ops[:, t], labels[:, t])
        return loss

    def forward(self, x, y, mask=None):
        ems = self.rnn(x)
        nllloss = self.crf(emissions=ems, tags=y, mask=mask)
        return nllloss

    def tag_sequence(self, x, mask=None):
        ems = self.rnn(x)
        scores, taggedseq = self.crf.decode(emissions=ems, mask=mask)
        return scores, taggedseq

class LSTMCRFExtractor(Extractor):
    def __init__(self, featscorer, model_file_name):
        super(LSTMCRFExtractor, self).__init__(featscorer)
        self.featscorer = featscorer
        self.net = self.load_model(filename=model_file_name)

    def load_model(self, filename):
        try:
            checkpt = torch.load("saved/" + filename)
            net = LSTM_CRF(inp_size=checkpt["config"]["inp_size"],
                           lstm_hid_size=checkpt["config"]["lstm_hid_size"],
                           num_layers=checkpt["config"]["num_layers"],
                           dropout=checkpt["config"]["dropout"])
            net.load_state_dict(checkpt['state_dict'])
        except:
            raise Exception("Error! Model checkpoint file not found, or file not saved correctly.")
        return net

    def extract_rationales(self, x):
        scoreop = self.featscorer.get_score_features(x)
        bertops = scoreop["bert_outputs"]
        attnscores = scoreop["attention_scores"]
        inpids = scoreop["tokenizer"]["input_ids"]
        maxcontiglen = torch.sum(scoreop["tokenizer"]["attention_mask"],dim=1)

        rationale_data = {}
        rationales = []
        rationalevecs = []

        self.net.eval()
        lstm_inp = torch.cat((bertops, attnscores), dim=2)
        scores, preds = self.net.tag_sequence(lstm_inp, mask=scoreop["tokenizer"]["attention_mask"]) # (batchsz, seqlen)
        for batch_item in range(0, bertops.shape[0]):
            i = 0
            while i < maxcontiglen[batch_item].item():
                if i==1:
                    j = i
                    while j < maxcontiglen[batch_item].item() and preds[batch_item][j]==1:
                        j += 1
                    rationales.append((batch_item, self.featscorer.decode_inputids(inpids[batch_item, i:j]), scores[batch_item]))
                    rationalevecs.append((batch_item, torch.mean(bertops[batch_item, i:j, :], dim=0), scores[batch_item]))
                    i = j-1
                i += 1

        rationale_data["rationales"] = rationales
        rationale_data["rationale_avg_vec"] = rationalevecs
        return rationale_data

def train_lstmcrf_model(train_x, dev_x, fsmodel, FILENAME, device):
    inp_size = 769
    lstm_hid_size = 256
    num_layers = 2
    dropout = 0.1

    net = LSTM_CRF(inp_size=inp_size, # 768 + 1 = 769 dim input vector
                     lstm_hid_size=lstm_hid_size,
                     num_layers=num_layers,
                     dropout=dropout)
    net = net.to(device)

    #fsmodel = FeatureImportanceScorer(model_file_name=fsmodel_filename)
    heuristicext = HeuristicExtractor(fsmodel)

    # hyperparameters
    EPOCHS = 20
    batch_size = 32
    lr = 2e-5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)

    for epoch in range(EPOCHS):
        tot_loss = 0.0
        perm = torch.randperm(len(train_x))
        for i in range(batch_size, len(perm), batch_size):
            optimizer.zero_grad()
            xbatch_str = [train_x[j] for j in perm[i - batch_size:i]]
            scoreop = fsmodel.get_score_features(xbatch_str)
            lstm_inp = torch.cat((scoreop["bert_outputs"], scoreop["attention_scores"]), dim=2)
            xbatch = lstm_inp.to(device)

            ybatch = heuristicext.extract_rationales(xbatch_str)["binary_mask"]
            ybatch = ybatch.to(device)

            mask = scoreop["tokenizer"]["attention_mask"]
            mask = mask[perm[i-batch_size:i]].to(device)

            crfloss = net(x = xbatch, y = ybatch, mask=mask)
            tot_loss += crfloss.item()
            crfloss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            del crfloss

        print("Epoch {0}      Loss {1}".format(epoch, tot_loss))
        if epoch % 4 == 0:
            #acc, f1 = evaluate_ext_model(model, dev_x, dev_y, device)
            checkpt = {
                'state_dict': net.state_dict(),
                'epochs': epoch,
                'batch_size': batch_size,
                'lr': lr,
                'config': {'inp_size': inp_size, 'lstm_hid_size':lstm_hid_size, 'num_layers':num_layers, 'dropout':dropout}}
            torch.save(checkpt, FILENAME[0:FILENAME.index('.')] + str(epoch) + 'epoch.pt')

    #acc, f1 = evaluate_ext_model(model, dev_x, dev_y, device)
    checkpt = {
        'state_dict': net.state_dict(),
        'epochs': EPOCHS,
        'batch_size': batch_size,
        'lr': lr,
        'config': {'inp_size': inp_size, 'lstm_hid_size':lstm_hid_size, 'num_layers':num_layers, 'dropout':dropout}}
    torch.save(checkpt, FILENAME)

# if __name__ == '__main__':
#     md = BiLSTMNetwork(10,20)
#     ls = md.mask_loss_hard(attns=torch.tensor([[0.5,0.5]], requires_grad=True),
#                       ops=torch.tensor([[[0.1,0.9],[0.1,0.9]]], requires_grad=True),
#                       contiglens=torch.tensor([2.0,2.0], requires_grad=True))
#     # lf = CustomLoss()
#     # ls = lf.forward(ops=torch.tensor([1.0,2.0,3.0,4.0,5.0], requires_grad=True))
#     ls.backward()