from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, GPT2Model
from sklearn.preprocessing import MultiLabelBinarizer
from mitnewsclassify2 import tfidf, tfidf_bi, download
import os
import gc
import gzip
import pickle
import csv

def print_f(*args):
    print(*args, flush=True)

print_f('All imports seem good!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_f('Using device:', device)

# tokenizing
train_size = None  # None for full dataset
MODEL = 'gpt2'
NR = 'first'


class EmbeddedDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], idx


print_f('Loading NYT dataset...')

# open the train data given to us by Max
with gzip.open('../data/NYTcorpus_train.p.gz', mode='r') as f:
    train_data = pickle.load(f)

print_f('Data loaded.')

# train and test data labels are coded in numbers,
# but the models predict human-readable labels,
# so we need to re-map these. 
# Let's use one of the files downloaded by the mitnewsclassify package
with open('../data/nyt-theme-tags.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    tags_dict = {row['tags_id']: row['tag'] for row in reader}

# extract actual article texts from data samples
train_articles = [d[2] for d in train_data] 

# map the number-coded labels to human-readable labels
train_labels_lists = [list(map(tags_dict.get, d[3:])) for d in train_data]

X_train, y_train = train_articles[:train_size], train_labels_lists[:train_size]

print_f('X_train', len(X_train))
print_f('y_train', len(y_train))

# start actual vectorization
dataset, output_path = X_train, y_train, f'/gpfs/space/projects/stud_nlp_share/kristjan/ensemble/embedded_train_FULL_ensemble'

####


chunk_size = 50_000


total_chunks = len(dataset) // chunk_size + 1
print_f('total chunks', total_chunks)

iterator = DataLoader(dataset, batch_size=chunk_size)
print_f(f'Vectorizing dataset for ', output_path)

X_train = []
chunk_id = 1

print_f('Starting at chunk id', chunk_id)

for i, batch in enumerate(iterator):
    inputs, attention_mask = batch

    real_batch_size = inputs.shape[0]

    inputs = inputs.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output = model(input_ids=inputs, attention_mask=attention_mask)

    output = output[0]

    # indices of last non-padded elements in each sequence
    # adopted from https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1290-L1302
    last_non_padded_ids = torch.ne(inputs, tokenizer.pad_token_id).sum(-1) - 1

    embeddings = output[range(real_batch_size), last_non_padded_ids, :]

    X_train += embeddings.detach().cpu()

    if len(X_train) >= chunk_size:
        saved_dataset = EmbeddedDataset(torch.stack(X_train))
        torch.save(saved_dataset, f'{output_path}_chunk{chunk_id}of{total_chunks}.pt', pickle_protocol=4)
        X_train = []
        chunk_id += 1

# take care of what's left after loop
if len(X_train) > 0:
    saved_dataset = EmbeddedDataset(torch.stack(X_train))
    torch.save(saved_dataset, f'{output_path}_chunk{chunk_id}of{total_chunks}.pt', pickle_protocol=4)
  
print_f('All done!')