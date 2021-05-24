import csv
import pickle
import gzip
from torch.utils.data import Dataset
import re
from collections import Counter
import gc
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


class GPTTokenizedDatasetWithoutLabels(Dataset):
    def __init__(self, articles, tokenizer):

        self.tokenizer = tokenizer
        
        print_f('GPTTokenizedDataset init - tokenizing...')
        articles = tokenizer(articles, add_special_tokens=True, padding="max_length", truncation=True,
                                       max_length=1024, return_tensors="pt", return_attention_mask=True)
        
        self.input_ids = articles['input_ids']
        self.attention_mask = articles['attention_mask']
        del articles
        gc.collect()
        print_f('GPTTokenizedDataset init - tokenizing done.')
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], None

    
class GPTTokenizedDataset(Dataset):
    def __init__(self, articles, labels, tokenizer):

        self.tokenizer = tokenizer
        
        print_f('GPTTokenizedDataset init - labels2vec...')
        self.labels = labels2vec(labels)
        del labels
        gc.collect()
        print_f('GPTTokenizedDataset init - labels2vec done.')
        
        print_f('GPTTokenizedDataset init - tokenizing...')
        articles = tokenizer(articles, add_special_tokens=True, padding="max_length", truncation=True,
                                       max_length=1024, return_tensors="pt", return_attention_mask=True)
        
        self.input_ids = articles['input_ids']
        self.attention_mask = articles['attention_mask']
        del articles
        gc.collect()
        print_f('GPTTokenizedDataset init - tokenizing done.')
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return self.articles[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

    
def load_nyt_articles(path, min_len=None, cutoff_tags=False):
    with gzip.open(path, mode='r') as f:
        data = pickle.load(f)
    print_f(f'Data loaded from {path}')

    # extract actual article texts from data samples
    articles = [d[2] for d in data]
    del data
    gc.collect()
    
    filtered_indices = range(len(articles))
    if min_len is not None:
        filtered_indices = [i for i, article in enumerate(articles) if len(article) >= min_len]
        articles = [articles[i] for i in filtered_indices]
        print_f('Articles after filtering:', len(articles))

    if cutoff_tags:
        articles = [remove_tags(a) for a in articles]
        
    gc.collect()
    return articles, filtered_indices


def load_nyt_data(min_len=None, cutoff_tags=False):
    # train and test data labels are coded in numbers,
    # but the models predict human-readable labels,
    # so we need to re-map these.
    # Let's use one of the files downloaded by the mitnewsclassify package
    with open('/gpfs/space/projects/stud_nlp_share/data/nyt-theme-tags.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        tags_dict = {row['tags_id']: row['tag'] for row in reader}

    # open the train data given to us by Max
    with gzip.open('/gpfs/space/projects/stud_nlp_share/data/NYTcorpus_train.p.gz', mode='r') as f:
        train_data = pickle.load(f)
    print_f('Train data loaded.')

    # extract actual article texts from data samples
    train_articles = [d[2] for d in train_data]
    # map the number-coded labels to human-readable labels
    train_labels_lists = [list(map(tags_dict.get, d[3:])) for d in train_data]
    del train_data
    gc.collect()
    
    # open the test data given to us by Max
    with gzip.open('/gpfs/space/projects/stud_nlp_share/data/NYTcorpus_test.p.gz', mode='r') as f:
        test_data = pickle.load(f)
    print_f('Test data loaded.')

    # extract actual article texts from data samples
    test_articles = [d[2] for d in test_data]
    # map the number-coded labels to human-readable labels
    test_labels_lists = [list(map(tags_dict.get, d[3:])) for d in test_data]
    del test_data
    gc.collect()
    
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
        
    gc.collect()
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


def labels2vec(labels):
    out2label = load_label_map('/gpfs/space/projects/stud_nlp_share/data/labels_dict_gpt.csv', '/gpfs/space/projects/stud_nlp_share/data/nyt-theme-tags.csv')
    mlb = MultiLabelBinarizer(classes=out2label)
    mlb.fit(out2label)
    return mlb.transform(labels).astype('uint8')


def vec2labels(vec):
    out2label = load_label_map('/gpfs/space/projects/stud_nlp_share/data/labels_dict_gpt.csv', '/gpfs/space/projects/stud_nlp_share/data/nyt-theme-tags.csv')
    mlb = MultiLabelBinarizer(classes=out2label)
    mlb.fit(out2label)
    return mlb.inverse_transform(vec)


def split_to_n_chunks(dataset, n=4):
    size = len(dataset)
    start_and_end_indices = [(size//n*(i-1), size//n*i) for i in range(1,n+1)]
    
    # take care of the end index of the last chunk in case dataset size did not divide by n
    start_and_end_indices[n-1] = (start_and_end_indices[n-1][0], size)

    chunks = []
    for start, end in start_and_end_indices:
        chunks.append(dataset[start:end])
    return chunks
    