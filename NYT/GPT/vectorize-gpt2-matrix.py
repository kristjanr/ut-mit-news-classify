import csv
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import csv
from tqdm import tqdm
import time
import gzip
import pickle
import os
import gc
import sys
import random

def print_f(*args):
    print(*args, flush=True)

print_f('All imports seem good!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_f('Using device:', device)

seed = 42
# tokenizing
train_size = None  # None for full dataset
test_size = None  # None for full dataset
cutoff_end_chars = None # None for full articles

tokenized_train_path = 'train_tokenized_FULL_matrix.pt'
tokenized_test_path = 'test_tokenized_FULL_matrix.pt'

# vectorizing
print('args', sys.argv)

MODEL = sys.argv[1] if len(sys.argv) >= 2 else 'gpt2'
batch_size = 8
chunk_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 5_000

print_f(f'Model: {MODEL} | Chunk size: {chunk_size} | Train limit: {train_size} | Test limit: {test_size} | Seed: {seed}')


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

print_f('Fitted MLB...')


# temporary dataset for storing tokenized articles & transformed labels
class NYTDataset(Dataset):
    def __init__(self, articles, labels):

        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        articles = self.tokenizer(articles, add_special_tokens=True, padding="max_length", truncation=True,
                                       max_length=1024, return_tensors="pt", return_attention_mask=True)

        self.input_ids = articles['input_ids']
        self.attention_mask = articles['attention_mask']

        self.labels = mlb.transform(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return self.articles[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

################
# LOAD DATA
################

print_f('Loading NYT dataset...')

# train and test data labels are coded in numbers,
# but the models predict human-readable labels,
# so we need to re-map these. 
# Let's use one of the files downloaded by the mitnewsclassify package
with open('../data/nyt-theme-tags.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    tags_dict = {row['tags_id']: row['tag'] for row in reader}

if not os.path.exists(tokenized_train_path):
    print_f('No tokenized training set found. Loading...')
    # open the train data given to us by Max
    with gzip.open('../data/NYTcorpus_train.p.gz', mode='r') as f:
        train_data = pickle.load(f)
    print_f('Data loaded.')

    random.Random(seed).shuffle(train_data)
    # extract actual article texts from data samples
    train_articles = [d[2][:cutoff_end_chars] for d in train_data] 
    # map the number-coded labels to human-readable labels
    train_labels_lists = [list(map(tags_dict.get, d[3:])) for d in train_data]

    print_f('Tokenizing dataset...')
    train_dataset = NYTDataset(train_articles[:train_size], train_labels_lists[:train_size])
    
    print_f('Saving tokenized dataset...')
    torch.save(train_dataset, tokenized_train_path, pickle_protocol=4)
else:
    print_f('Found tokenized training set...')
    train_dataset = torch.load(tokenized_train_path)


if not os.path.exists(tokenized_test_path):
    print_f('No tokenized test set found. Loading...')
    # open the test data given to us by Max
    with gzip.open('../data/NYTcorpus_test.p.gz', mode='r') as f:
        test_data = pickle.load(f)
    print_f('Data loaded.')
    
    random.Random(seed).shuffle(test_data)
    # extract actual article texts from data samples
    test_articles = [d[2][:cutoff_end_chars] for d in test_data]
    # map the number-coded labels to human-readable labels
    test_labels_lists = [list(map(tags_dict.get, d[3:])) for d in test_data]

    print_f('Tokenizing dataset...')
    test_dataset = NYTDataset(test_articles[:test_size], test_labels_lists[:test_size])
    
    print_f('Saving tokenized dataset...')
    torch.save(test_dataset, tokenized_test_path, pickle_protocol=4)
else:
    print_f('Found tokenized test set...')
    test_dataset = torch.load(tokenized_test_path)


# start actual vectorization with GPT2
runs = [(train_dataset, f'vectorized-matrix/embedded_matrix_train_FULL_{MODEL.replace("-", "_")}'), (test_dataset, f'vectorized-matrix/embedded_matrix_test_FULL_{MODEL.replace("-", "_")}')]

print_f('Loading model...')
model = GPT2Model.from_pretrained(MODEL)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(test_dataset.tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)

for dataset, output_path in runs:

    total_chunks = len(dataset) // chunk_size + 1
    print_f('total chunks', total_chunks)

    # skip already embedded articles
    skip_n_articles = 0
    chunk_paths = sorted([chunk_path for chunk_path in os.listdir('.') if f'{output_path}_chunk' in chunk_path])

    print_f('chunks', chunk_paths)

    if len(chunk_paths) > 0:
        for i, chunk_path in enumerate(chunk_paths):
            chunk = torch.load(chunk_path)

            skip_n_articles += len(chunk)
            print_f(f'Chunk at "{chunk_path}" has {len(chunk)} articles.')

            del chunk
            gc.collect()

        print_f('skip:', skip_n_articles)

        if skip_n_articles >= len(dataset):
            print_f('Looks like the dataset is fully embedded already. Skipping this dataset...')
            continue

        print_f('dataset original', len(dataset))

        dataset.input_ids = dataset.input_ids[skip_n_articles:]
        dataset.attention_mask = dataset.attention_mask[skip_n_articles:]
        dataset.labels = dataset.labels[skip_n_articles:]

        print_f('dataset after skipping', len(dataset))

    iterator = DataLoader(dataset, batch_size=batch_size)
    print_f(f'Vectorizing dataset for ', output_path)

    X_train = []
    y_train = []
    chunk_id = len(chunk_paths) + 1

    print_f('Starting at chunk id', chunk_id)

    for i, batch in enumerate(tqdm(iterator)):
        inputs, attention_mask, labels = batch

        real_batch_size = inputs.shape[0]

        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = torch.tensor(labels).to(device)

        with torch.no_grad():
            output = model(input_ids=inputs, attention_mask=attention_mask)

        embeddings = output[0] # (batch_size, window_size, hidden_dim), e.g. (8, 1024, 768)

        X_train += embeddings.detach().cpu()
        y_train += labels.detach().cpu()

        if len(X_train) >= chunk_size:
            print_f(f'Saving chunk: {output_path}_chunk{chunk_id}of{total_chunks}.pt')
            saved_dataset = GPTEmbeddedDataset(torch.stack(X_train), torch.stack(y_train))
            torch.save(saved_dataset, f'{output_path}_chunk{chunk_id}of{total_chunks}.pt', pickle_protocol=4)
            chunk_id += 1

            del saved_dataset
            del X_train
            del y_train
            gc.collect()
            X_train = []
            y_train = []

    # take care of what's left after loop
    if len(X_train) >= 0:
        print_f(f'Saving chunk: {output_path}_chunk{chunk_id}of{total_chunks}.pt')
        saved_dataset = GPTEmbeddedDataset(torch.stack(X_train), torch.stack(y_train))
        torch.save(saved_dataset, f'{output_path}_chunk{chunk_id}of{total_chunks}.pt', pickle_protocol=4)

        del saved_dataset
        del X_train
        del y_train
        gc.collect()
  
print_f('All done!')