import csv
import pickle
import gzip
from torch.utils.data import Dataset
import re
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer


capital_tag_pattern = '([A-Z]{3,}(\s|$))+'


def print_f(*args):
    print(*args, flush=True)


def remove_tags(text, end_start=-100):
    return text[:end_start] + re.sub(capital_tag_pattern, '', text[end_start:])


class GPTVectorizedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


class GPTTokenizedDataset(Dataset):
    def __init__(self, articles, labels, tokenizer):

        self.tokenizer = tokenizer

        articles = tokenizer(articles, add_special_tokens=True, padding="max_length", truncation=True,
                                       max_length=1024, return_tensors="pt", return_attention_mask=True)

        self.input_ids = articles['input_ids']
        self.attention_mask = articles['attention_mask']

        self.labels = labels2vec(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return self.articles[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def load_nyt_data(min_len=None, cutoff_tags=False):
    # train and test data labels are coded in numbers,
    # but the models predict human-readable labels,
    # so we need to re-map these.
    # Let's use one of the files downloaded by the mitnewsclassify package
    with open('../data/nyt-theme-tags.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        tags_dict = {row['tags_id']: row['tag'] for row in reader}

    # open the train data given to us by Max
    with gzip.open('../data/NYTcorpus_train.p.gz', mode='r') as f:
        train_data = pickle.load(f)
    print_f('Train data loaded.')

    # extract actual article texts from data samples
    train_articles = [d[2] for d in train_data]
    # map the number-coded labels to human-readable labels
    train_labels_lists = [list(map(tags_dict.get, d[3:])) for d in train_data]

    # open the test data given to us by Max
    with gzip.open('../data/NYTcorpus_test.p.gz', mode='r') as f:
        test_data = pickle.load(f)
    print_f('Test data loaded.')

    # extract actual article texts from data samples
    test_articles = [d[2] for d in test_data]
    # map the number-coded labels to human-readable labels
    test_labels_lists = [list(map(tags_dict.get, d[3:])) for d in test_data]

    if min_len is not None:
        filtered_train_indices = [i for i, article in enumerate(train_articles) if len(article) >= min_len]
        train_articles = [train_articles[i] for i in filtered_train_indices]
        train_labels_lists = [train_labels_lists[i] for i in filtered_train_indices]
        print_f('Train articles after filtering:', len(train_articles))

        filtered_test_indices = [i for i, article in enumerate(test_articles) if len(article) >= min_len]
        test_articles = [test_articles[i] for i in filtered_test_indices]
        test_labels_lists = [test_labels_lists[i] for i in filtered_test_indices]
        print_f('Test articles after filtering:', len(test_articles))

    if cutoff_tags:
        train_articles = [remove_tags(a) for a in train_articles]
        test_articles = [remove_tags(a) for a in test_articles]

    return train_articles, train_labels_lists, test_articles, test_labels_lists


def loadcsv(filename):
    with open(filename, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def load_label_map(out2id_path, id2label_path):

    out2id = loadcsv(out2id_path)
    out2id = {int(row[0]): row[1] for row in out2id}

    id2label_raw = loadcsv(id2label_path)
    id2label = {}

    for row in id2label_raw:
        if row == []:
            continue
        id2label[row[1]] = row[2]

    out2label = [id2label[out2id[out]] for out in sorted(out2id.keys())]

    return out2label


out2label = load_label_map('../data/labels_dict_gpt.csv', '../data/nyt-theme-tags.csv')
mlb = MultiLabelBinarizer(classes=out2label)
mlb.fit(out2label)


def labels2vec(labels):
    return mlb.transform(labels)


def vec2labels(vec):
    return mlb.inverse_transform(vec)