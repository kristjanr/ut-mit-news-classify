import os
import sys
module_path = "/gpfs/space/home/roosild/ut-mit-news-classify/NYT/"
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import DataLoader
import torch
import os
from tqdm.auto import tqdm
from utils import print_f, load_nyt_articles, split_to_n_chunks
import gc



print_f('All imports seem good!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_f('Using device:', device)


########################
# LOAD TRAIN DATA
########################

TOTAL_NR_OF_CHUNKS = 4

## change this for each chunk
NR = 4

## change this for non-cutoff data
cutoff_end_chars = True 


print_f('Loading NYT train dataset...')

train_articles, _ = load_nyt_articles('/gpfs/space/projects/stud_nlp_share/data/NYTcorpus_train.p.gz', min_len=500, cutoff_tags=cutoff_end_chars)

print_f(f'Splitting to {TOTAL_NR_OF_CHUNKS} chunks and processing chunk nr {NR}')

size_str = f'{len(train_articles)//1000}k'

train_articles = split_to_n_chunks(train_articles, TOTAL_NR_OF_CHUNKS)[NR-1]

gc.collect()

print_f(f'There are {len(train_articles)} articles in this chunk')

if cutoff_end_chars:
    ending = 'min500_cutoff_replace'
else:
    ending = 'min500_complete'

os.makedirs('tokenized', exist_ok=True)

train_path = f'/gpfs/space/projects/stud_nlp_share/cutoff/GPT/tokenized/train_{size_str}_{ending}_chunk{NR}of{TOTAL_NR_OF_CHUNKS}.pt'

print('Will save train to:', train_path)


########################
# TOKENIZE TRAIN DATA
########################
from utils import GPTTokenizedDatasetWithoutLabels, load_nyt_data
from transformers import GPT2TokenizerFast

MODEL = 'gpt2'

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

import time

tic = time.perf_counter()

if not os.path.exists(train_path):
    print_f('Tokenizing train dataset...')
    train_dataset = GPTTokenizedDatasetWithoutLabels(train_articles, tokenizer)
    del train_articles
    gc.collect()
    
    print_f('Saving tokenized dataset...')
    torch.save(train_dataset, train_path, pickle_protocol=4)
    
    del train_dataset
    gc.collect()
    
else:
    print_f('Found tokenized training set...')
    train_dataset = torch.load(train_path)


toc = time.perf_counter()
print_f(f'Done train tokenization part in {toc - tic:0.4f} seconds!')


########################
# LOAD TEST DATA
########################


print_f('Loading NYT test dataset...')

test_articles, _ = load_nyt_articles('/gpfs/space/projects/stud_nlp_share/data/NYTcorpus_test.p.gz', min_len=500, cutoff_tags=cutoff_end_chars)

print_f(f'There are {len(test_articles)} articles')



########################
# TOKENIZE ALL TEST DATA
########################

test_path = f'/gpfs/space/projects/stud_nlp_share/cutoff/GPT/tokenized/test_{size_str}_{ending}_chunk1of1.pt'
print('Will save test to:', test_path)

import time

tic = time.perf_counter()


if not os.path.exists(test_path):
    print_f('Tokenizing test dataset...')
    test_dataset = GPTTokenizedDatasetWithoutLabels(test_articles, tokenizer)
    del test_articles
    gc.collect()
    
    print_f('Saving tokenized dataset...')
    torch.save(test_dataset, test_path, pickle_protocol=4)
    del test_dataset
    gc.collect()
else:
    print_f('Found tokenized training set...')
    test_dataset = torch.load(test_path)


toc = time.perf_counter()
print_f(f'Done test tokenization part in {toc - tic:0.4f} seconds!')


