import os
import sys
module_path = "/gpfs/space/home/roosild/ut-mit-news-classify/NYT/"
if module_path not in sys.path:
    sys.path.append(module_path)
from torch.utils.data import DataLoader
import torch
import gc
from tqdm.auto import tqdm
from utils import print_f, GPTVectorizedDataset
from transformers import GPT2Model

print_f('All imports seem good!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_f('Using device:', device)

import time

tic = time.perf_counter()

MODEL = 'gpt2'
batch_size = 16

# change this for each chunk
NR = 2
cutoff_end_chars = True 


import os
tokenized_directory = '/gpfs/space/projects/stud_nlp_share/cutoff/GPT/tokenized/'

def is_correct_chunk(filename):
    if cutoff_end_chars and 'cutoff' in filename or not cutoff_end_chars and 'complete' in filename:
        chunk_nr = filename.split('chunk')[1].split('of')[0]
        return NR == int(chunk_nr)
    return False

filenames = os.listdir(tokenized_directory)

for filename in filenames:
    if is_correct_chunk(filename):
        if 'train' in filename:
            tokenized_train_filename = filename
        else:
            tokenized_test_filename = filename
        
tokenized_train_path = tokenized_directory + tokenized_train_filename
tokenized_test_path = tokenized_directory + tokenized_test_filename

os.makedirs('vectorized', exist_ok=True)

vectorized_train_path = f'/gpfs/space/projects/stud_nlp_share/cutoff/GPT/vectorized/{tokenized_train_filename}'
vectorized_test_path = f'/gpfs/space/projects/stud_nlp_share/cutoff/GPT/vectorized/{tokenized_test_filename}'

# start actual vectorization with GPT2
runs = [(tokenized_train_path, vectorized_train_path), (tokenized_test_path, vectorized_test_path)]

print_f('Loading model...')
model = GPT2Model.from_pretrained(MODEL)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(torch.load(tokenized_test_path).tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)

for dataset_path, output_path in runs:
    
    print_f('Loading tokenized dataset...')
    dataset = torch.load(dataset_path)
    print_f(f'Loaded {dataset_path}')
    
    iterator = DataLoader(dataset, batch_size=batch_size)
    print_f('Vectorizing dataset for ', output_path)

    X_train = []
    y_train = []
    gc.collect()

    for i, batch in enumerate(tqdm(iterator)):
        inputs, attention_mask, labels = batch

        real_batch_size = inputs.shape[0]

        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = torch.tensor(labels).to(device)

        with torch.no_grad():
            output = model(input_ids=inputs, attention_mask=attention_mask)

        output = output[0]

        # indices of last non-padded elements in each sequence
        # adopted from https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1290-L1302
        last_non_padded_ids = torch.ne(inputs, dataset.tokenizer.pad_token_id).sum(-1) - 1

        embeddings = output[range(real_batch_size), last_non_padded_ids, :]

        X_train += embeddings.detach().cpu()
        y_train += labels.detach().cpu()
        del output
        del embeddings
        gc.collect()

    print_f('Saving:', output_path)
    saved_dataset = GPTVectorizedDataset(torch.stack(X_train), torch.stack(y_train))
    torch.save(saved_dataset, output_path, pickle_protocol=4)
    del saved_dataset           
    del embeddings
    del dataset
    gc.collect()
    
toc = time.perf_counter()
print_f(f'Done in {toc - tic:0.4f} seconds!')