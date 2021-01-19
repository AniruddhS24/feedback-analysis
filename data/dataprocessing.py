import os
import pandas as pd
import re

# https://www.kaggle.com/ilhamfp31/yelp-review-dataset
# https://www.kaggle.com/kazanova/sentiment140
# most preprocessing is taken care of my bert tokenizer... implement only if needed

def read_dataset(dataset, num_samples=10000, min_ex_length=100):
    if dataset == 'TWITTER':
        return read_twitter(path='data/twitterreviews.csv', num_samples=num_samples, min_ex_length=min_ex_length)
    if dataset == 'YELP':
        return read_yelp(path='data/yelpreviews.csv', num_samples=num_samples, min_ex_length=min_ex_length)
    if dataset == 'SST':
        return read_sst(path='data/stanfordSentimentTreebank', num_samples=num_samples, min_ex_length=min_ex_length)

def read_twitter(path, num_samples, min_ex_length):
    db = pd.read_csv(path, encoding='latin-1', header=None)
    db.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    data_x = []
    data_y = []
    for i in range(min(num_samples, len(db.index))):
        if len(db['text'].loc[i]) >= min_ex_length:
            text = db['text'].loc[i]
            # remove URLs
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"www\.\S+", "", text)
            # remove @ mentions
            text = re.sub(r"@[\w]*", "", text)
            # remove punctuation
            # text = re.sub("[^a-zA-Z#]", " ", text)
            data_x.append(text)
            lbl = db['target'].loc[i]
            if lbl==0:
                data_y.append(0)
            else:
                data_y.append(1)
    return split_train_dev(data_x, data_y)

def read_yelp(path, num_samples, min_ex_length):
    db = pd.read_csv(path, header=None, error_bad_lines=False, engine="python")
    db.columns = ['target', 'text']
    data_x = []
    data_y = []
    for i in range(min(num_samples, len(db.index))):
        if len(db['text'].loc[i]) >= min_ex_length:
            text = db['text'].loc[i]
            # remove URLs
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"www\.\S+", "", text)
            # remove newline
            text = re.sub(r"\\n+", "", text)
            # remove weird characters
            text = re.sub(r"[^a-zA-Z\s#.,?/\'\-!]", " ", text)
            data_x.append(text)
            lbl = db['target'].loc[i]
            if lbl==1:
                data_y.append(0)
            else:
                data_y.append(1)
    return split_train_dev(data_x, data_y)

def read_sst(path, num_samples, min_ex_length):
    dictionary = pd.read_csv(os.path.join(path, "dictionary.txt"), sep="|")
    dictionary.columns = ["phrase", "phrase_id"]
    dictionary = dictionary.set_index("phrase_id")

    phrase_sent = pd.read_csv(os.path.join(path, "sentiment_labels.txt"), sep="|")
    phrase_sent.columns = ["phrase_id", "sentiment"]
    phrase_sent = phrase_sent.set_index("phrase_id")

    phrase_dataset = dictionary.join(phrase_sent)
    data_x = phrase_dataset["phrase"][:min(num_samples, len(phrase_dataset.index))].tolist()
    data_y = pd.cut(phrase_dataset["sentiment"][:min(num_samples, len(phrase_dataset.index))],
                    bins=[0.0,0.2,0.4,0.6,0.8, 1.0],
                    include_lowest=True,
                    labels=[0,1,2,3,4]).tolist()

    return split_train_dev(data_x, data_y)

def split_train_dev(data_x, data_y, p=0.8):
    splidx = round(len(data_y)*p)
    return data_x[0:splidx], data_y[0:splidx], data_x[splidx:], data_y[splidx:]

if __name__ == '__main__':
    train_x, train_y, dev_x, dev_y = read_dataset('YELP', 10)
    for i in range(len(train_x)):
        print(train_y[i], train_x[i])
