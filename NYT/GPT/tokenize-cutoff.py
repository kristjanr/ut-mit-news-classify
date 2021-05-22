import os
import sys
module_path = "/gpfs/space/home/roosild/ut-mit-news-classify/NYT/utils.py"
if module_path not in sys.path:
    sys.path.append(module_path)
from torch.utils.data import DataLoader
import torch
import os
from tqdm.auto import tqdm
from utils import print_f, load_nyt_data, split_to_n_chunks


print_f('All imports seem good!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_f('Using device:', device)

cutoff_end_chars = True 



########################
# LOAD DATA
########################

TOTAL_NR_OF_CHUNKS = 4000

## change this for each chunk
NR = 1


print_f('Loading NYT dataset...')

train_articles, train_labels_lists, test_articles, test_labels_lists = load_nyt_data(min_len=500, cutoff_tags=cutoff_end_chars)

print_f('Splitting to 4 chunks and processing chunk nr {NR}')


train_articles = split_to_n_chunks(train_articles, TOTAL_NR_OF_CHUNKS)[NR-1]
train_labels_lists = split_to_n_chunks(train_labels_lists, TOTAL_NR_OF_CHUNKS)[NR-1]
test_articles = split_to_n_chunks(test_articles, TOTAL_NR_OF_CHUNKS)[NR-1]
test_labels_lists = split_to_n_chunks(test_labels_lists, TOTAL_NR_OF_CHUNKS)[NR-1]

size_str = f'{len(train_articles)//1000}k'

if cutoff_end_chars:
    ending = 'min500_cutoff_replace'
else:
    ending = 'min500_complete'

os.makedirs('tokenized', exist_ok=True)

train_path = f'/gpfs/space/projects/stud_nlp_share/cutoff/GPT/tokenized/train_{size_str}_{ending}_chunk{NR}of{TOTAL_NR_OF_CHUNKS}.pt'
test_path = f'/gpfs/space/projects/stud_nlp_share/cutoff/GPT/tokenized/test_{size_str}_{ending}_chunk{NR}of{TOTAL_NR_OF_CHUNKS}.pt'

print('Will save train to:', train_path)
print('Will save test to:', test_path)

########################
# TOKENIZE DATA
########################
from utils import GPTTokenizedDataset, load_nyt_data
from transformers import GPT2TokenizerFast

MODEL = 'gpt2'

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

if not os.path.exists(train_path):
    print_f('Tokenizing train dataset...')
    train_dataset = GPTTokenizedDataset(train_articles, train_labels_lists, tokenizer)
    
    print_f('Saving tokenized dataset...')
    torch.save(train_dataset, train_path, pickle_protocol=4)
else:
    print_f('Found tokenized training set...')
    train_dataset = torch.load(train_path)


if not os.path.exists(test_path):
    print_f('Tokenizing test dataset...')
    test_dataset = GPTTokenizedDataset(test_articles, test_labels_lists, tokenizer)
    
    print_f('Saving tokenized dataset...')
    torch.save(test_dataset, test_path, pickle_protocol=4)
else:
    print_f('Found tokenized training set...')
    test_dataset = torch.load(test_path)