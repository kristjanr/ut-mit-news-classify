import os
import sys
module_path = "/gpfs/space/home/mykyta/nlp/ut-mit-news-classify/NYT/"
if module_path not in sys.path:
    sys.path.append(module_path)
from torch.utils.data import DataLoader
import torch
import os
from tqdm.auto import tqdm
from utils import print_f, load_nyt_data


print_f('All imports seem good!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_f('Using device:', device)

cutoff_end_chars = True 

train_size = 150_000  # None for full dataset
test_size = 15_000  # None for full dataset
size_str = f'{train_size//1000}k'

if cutoff_end_chars:
    ending = 'min500_cutoff_replace'
else:
    ending = 'min500_complete'

os.makedirs('tokenized', exist_ok=True)

train_path = f'tokenized/train_{size_str}_{ending}.pt'
test_path = f'tokenized/test_{size_str}_{ending}.pt'

print('Will save train to:', train_path)
print('Will save test to:', test_path)

########################
# LOAD & TOKENIZE DATA
########################
from utils import GPTTokenizedDataset, load_nyt_data
from transformers import GPT2Tokenizer

MODEL = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

print_f('Loading NYT dataset...')

train_articles, train_labels_lists, test_articles, test_labels_lists = load_nyt_data(min_len=500, cutoff_tags=cutoff_end_chars)

if not os.path.exists(train_path):
    print_f('Tokenizing train dataset...')
    train_dataset = GPTTokenizedDataset(train_articles[:train_size], train_labels_lists[:train_size], tokenizer)
    
    print_f('Saving tokenized dataset...')
    torch.save(train_dataset, train_path, pickle_protocol=4)
else:
    print_f('Found tokenized training set...')
    train_dataset = torch.load(train_path)


if not os.path.exists(test_path):
    print_f('Tokenizing test dataset...')
    test_dataset = GPTTokenizedDataset(test_articles[:test_size], test_labels_lists[:test_size], tokenizer)
    
    print_f('Saving tokenized dataset...')
    torch.save(test_dataset, test_path, pickle_protocol=4)
else:
    print_f('Found tokenized training set...')
    test_dataset = torch.load(test_path)