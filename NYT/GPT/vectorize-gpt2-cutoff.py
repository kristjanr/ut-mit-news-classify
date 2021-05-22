import os
import sys
module_path = "/gpfs/space/home/mykyta/nlp/ut-mit-news-classify/NYT/"
if module_path not in sys.path:
    sys.path.append(module_path)
from torch.utils.data import DataLoader
import torch
import os
from tqdm.auto import tqdm
from utils import print_f, GPTVectorizedDataset
from transformers import GPT2Model

print_f('All imports seem good!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_f('Using device:', device)


MODEL = 'gpt2'
batch_size = 16
chunk_size = 200_000

tokenized_train_path = f'tokenized/train_150k_min500_complete.pt'
tokenized_test_path = f'tokenized/test_150k_min500_complete.pt'

os.makedirs('vectorized', exist_ok=True)

vectorized_train_path = f'vectorized/train_150k_min500_complete_slurm.pt'
vectorized_test_path = f'vectorized/test_150k_min500_complete_slurm.pt'

print_f('Loading vectorized dataset...')

train_dataset = torch.load(tokenized_train_path)
test_dataset = torch.load(tokenized_test_path)

# start actual vectorization with GPT2
runs = [(train_dataset, vectorized_train_path), (test_dataset, vectorized_test_path)]

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
            print_f('Looks like the dataset if fully embedded already. Skipping this dataset...')
            continue

        print_f('dataset original', len(dataset))

        dataset.input_ids = dataset.input_ids[skip_n_articles:]
        dataset.attention_mask = dataset.attention_mask[skip_n_articles:]
        dataset.labels = dataset.labels[skip_n_articles:]

        print_f('dataset after skipping', len(dataset))

    iterator = DataLoader(dataset, batch_size=batch_size)
    print_f('Vectorizing dataset for ', output_path)

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

        output = output[0]

        # indices of last non-padded elements in each sequence
        # adopted from https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1290-L1302
        last_non_padded_ids = torch.ne(inputs, dataset.tokenizer.pad_token_id).sum(-1) - 1

        embeddings = output[range(real_batch_size), last_non_padded_ids, :]

        X_train += embeddings.detach().cpu()
        y_train += labels.detach().cpu()

        if len(X_train) >= chunk_size:
            print_f('Saving chunk:', output_path)
            saved_dataset = GPTVectorizedDataset(torch.stack(X_train), torch.stack(y_train))
            torch.save(saved_dataset, output_path, pickle_protocol=4)
            X_train = []
            y_train = []
            chunk_id += 1

    # take care of what's left after loop
    if len(X_train) >= 0:
        print_f('Saving chunk:', output_path)
        saved_dataset = GPTVectorizedDataset(torch.stack(X_train), torch.stack(y_train))
        torch.save(saved_dataset, output_path, pickle_protocol=4)

print_f('All done!')