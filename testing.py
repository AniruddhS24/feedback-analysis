from models.featurescorer import *
from models.extractor import *

def main():
    fts = FeatureImportanceScorer('suppmodeel.pt')
    ext = HeuristicExtractor(fts, rationalelengthprop=0.15, num_rationales=2)
    ratdata = ext.extract_rationales(x=['This place is absolute garbage... Half of the tees are not available, including all the grass tees',
                                                    'I’m a professional makeup artist and I’ve been in search of brushes that are good quality but not super expensive and I came across these and I’m pleasantly surprised! I usually have multiple clients back to back so sometimes I don’t have time to clean my brushes so I needed some backups and now I find myself using these just as much as my Morphe and Real techniques ones (which I love). The bristles are super soft and they don’t shed at all. The face brushes are super dense and don’t soak up too much product. The eye brushes are soft and can blend and pack on shadow really well. I also didn’t notice any odd smells. I would definitely reccomend this!',
                                                    'I purchased the 6 piece set to this a couple years ago. I have other brands of brushes like elf studio professional and Morphe studio professional and that 6 piece set I bought is my fav. The bristles were so soft and dense and fluffy. So I Came back to purchase the 16 piece set. The bristles are stiffer and less dense. I\'ll not be purchasing this again. It was disappointing.'])
    for item in ratdata["rationales"]:
        print(item[1])

def _compute_log_partition(emissions, mask, transitions):
    """Compute the partition function in log-space using the forward-algorithm.
    Args:
        emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
        mask (Torch.FloatTensor): (batch_size, seq_len)
    Returns:
        torch.Tensor: the partition scores for each batch.
            Shape of (batch_size,)
    """
    batch_size, seq_length, nb_labels = emissions.shape

    # in the first iteration, BOS will have all the scores
    alphas = torch.zeros(batch_size, nb_labels)

    for i in range(0, seq_length):
        # (bs, nb_labels) -> (bs, 1, nb_labels)
        e_scores = emissions[:, i].unsqueeze(1)

        # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
        t_scores = transitions.unsqueeze(0)

        # (bs, nb_labels)  -> (bs, nb_labels, 1)
        a_scores = alphas.unsqueeze(2)
        scores = e_scores + t_scores + a_scores
        new_alphas = torch.logsumexp(scores, dim=1)

        # set alphas if the mask is valid, otherwise keep the current values
        is_valid = mask[:, i].unsqueeze(-1)
        alphas = is_valid * new_alphas + (1 - is_valid) * alphas

    # add the scores for the final transition
    end_scores = alphas
    # return a *log* of sums of exps
    return torch.logsumexp(end_scores, dim=1)

def compute_partition(emissions, mask, transitions):
    batch_size, seqlen, num_tags = emissions.shape
    dp = torch.zeros(batch_size, num_tags)  # [B, C]
    trans = transitions.unsqueeze(0)  # [1, C, C]
    for t in range(seqlen):
        mask_t = mask[:, t].unsqueeze(1) # [B, 1]
        emit_t = emissions[:, t].unsqueeze(2)  # [B, C, 1]
        # add emissions and transitions to previous scores
        #print(dp.unsqueeze(1))
        dp_t = dp.unsqueeze(1) + emit_t + trans  # [B, 1, C] -> [B, C, C]
        #print(dp_t)
        dp_t = torch.logsumexp(dp_t, dim=2)  # [B, C, C] -> [B, C]
        #print(dp_t)
        # update score
        dp = dp_t * mask_t + dp * (1 - mask_t)
    return torch.logsumexp(dp, dim=1)  # partition function return (batchsz,) tensor

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tkn = tokenizer(text=["sample text testing itall"],
                     add_special_tokens=True,
                     padding='max_length',
                     truncation='only_first',
                     return_attention_mask=True,
                     return_tensors='pt')
