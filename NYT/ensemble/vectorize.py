import gzip
import pickle
import random
import csv
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import csv
from tqdm import tqdm
from mitnewsclassify2 import tfidf, tfidf_bi, download
import gc
import torch
import sys
import time
import os


def print_f(*args):
    print(*args, flush=True)

print_f('All imports seem good!')

seed = 42

n_chunks = 25

train_size = None
test_size = None

# print_f('Downloading mitwnewsclassify stuff...')
# download.download('tfidf')
# download.download('tfidf_bi')

# print_f('Flushing the buffer to let logs from package appear...')
# sys.stdout.flush()

class GPTEmbeddedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


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

print_f('Loading data...')

# open the train data given to us by Max
with gzip.open('../data/NYTcorpus_train.p.gz', mode='r') as f:
    train_data = pickle.load(f)

# open the test data given to us by Max
with gzip.open('../data/NYTcorpus_test.p.gz', mode='r') as f:
    test_data = pickle.load(f)

print_f('Data loaded.')

# shuffle just in case the test and train data were not shuffled before - 
# we will only measure model's accuracy on a few thousand samples
random.Random(seed).shuffle(train_data)
random.Random(seed).shuffle(test_data)

# train and test data labels are coded in numbers,
# but the models predict human-readable labels,
# so we need to re-map these. 
# Let's use one of the files downloaded by the mitnewsclassify package
with open('../data/nyt-theme-tags.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    tags_dict = {row['tags_id']: row['tag'] for row in reader}

# extract actual article texts from data samples
train_articles = [d[2] for d in train_data] 
test_articles = [d[2] for d in test_data]

# map the number-coded labels to human-readable labels
train_labels_lists = [list(map(tags_dict.get, d[3:])) for d in train_data]
test_labels_lists = [list(map(tags_dict.get, d[3:])) for d in test_data]


X_train, y_train = train_articles[:train_size], train_labels_lists[:train_size]
X_test, y_test = test_articles[:test_size], test_labels_lists[:test_size]

print_f('X_train', len(X_train))
print_f('y_train', len(y_train))
print_f('X_test', len(X_test))
print_f('y_test', len(y_test))

runs = [(X_train, y_train, 'embedded/embedded_train_FULL_tfidf'), (X_test, y_test, 'embedded/embedded_test_FULL_tfidf')]

for X, y, output_path in runs:

    # skip already embedded articles
    chunk_paths = sorted([chunk_path for chunk_path in os.listdir('.') if f'{output_path}_chunk' in chunk_path])
    skip_n_chunks = len(chunk_paths)

    print_f('Chunks processed already:', skip_n_chunks)

    split_indices = np.array_split(np.arange(len(X)), n_chunks)

    print_f(f'Splitting into {len(split_indices)} chunks')

    for i, indices in enumerate(split_indices):

      if i < skip_n_chunks:
        continue

      chunk_path = f'{output_path}_chunk{i+1}of{len(split_indices)}.pt'
      print_f(f'Preprocessing dataset for ', chunk_path)

      X_chunk = X[indices[0]:indices[-1]]
      y_chunk = y[indices[0]:indices[-1]]

      print('Chunk size:', len(X_chunk))

      start = time.time()
      tfidf_vec = tfidf.getfeatures(X_chunk)
      tfidf_bi_vec = tfidf_bi.getfeatures(X_chunk)
      X_embedded = np.concatenate((tfidf_vec, tfidf_bi_vec), axis=1)

      y_embedded = mlb.transform(y_chunk)

      saved_dataset = GPTEmbeddedDataset(torch.tensor(X_embedded), torch.tensor(y_embedded))
      torch.save(saved_dataset, chunk_path, pickle_protocol=4)

      print_f(f'Time taken: {int(time.time() - start)/60:.1f}min')

      print_f()

      del tfidf_vec
      del tfidf_bi_vec
      del X_embedded
      del y_embedded
      del saved_dataset
      gc.collect()

    del split_indices
    gc.collect()


print_f('Done!')