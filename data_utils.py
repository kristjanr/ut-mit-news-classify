from sklearn.model_selection import train_test_split
import gzip
import pandas as pd
import numpy as np
import re
import math
import os

TEST_SIZE = 0.1

new_lines = re.compile(r'(\s+)|(\n+)')


def clean_opinion(remove_garbage):
    def cleaner_func(article):
        if type(article) != str:
            return ''

        # clean garbage
        if remove_garbage:
            if 'Today’s Headlines' in article:
                return ''

        return re.sub(new_lines, ' ', article)

    return cleaner_func


def load_opinion_data(val_size=None, n_train=10_000, n_val=10_000, remove_garbage=True, seed=42):
    '''
      If supplied, `val_size` [0, 1) - overrides n_train and n_val.
    '''

    assert os.path.exists('../articlesXXXXXXXX_wapo_all_opinion.tsv.gz'), 'Should download the dataset and place it in parent folder. '
    assert os.path.exists('../articlesXXXXXXXX_wapo_all_nopinion.tsv.gz'), 'Should download the dataset and place it in parent folder. '

    with gzip.open('../articlesXXXXXXXX_wapo_all_opinion.tsv.gz', mode='rt') as f:
        dfo = pd.read_csv(f, names=range(11), delimiter='\t')

    with gzip.open('../articlesXXXXXXXX_wapo_all_nopinion.tsv.gz', mode='rt') as f:
        dfno = pd.read_csv(f, names=range(11), delimiter='\t')

    opinion = dfo[6].map(clean_opinion(remove_garbage))
    nopinion = dfno[6].map(clean_opinion(remove_garbage))

    opinion = opinion[opinion != '']
    nopinion = nopinion[nopinion != '']

    opinion_labels = np.ones(opinion.shape)
    nopinion_labels = np.zeros(nopinion.shape)

    articles = np.concatenate((np.array(opinion), np.array(nopinion)))
    labels = np.concatenate((opinion_labels, nopinion_labels))

    # first, split test set from the rest
    n_test = math.floor(TEST_SIZE * len(articles))
    n_train_val = len(articles) - n_test
    x, x_test, y, y_test = train_test_split(articles, labels, train_size=n_train_val, test_size=n_test, random_state=seed)

    # split train & validation
    if val_size is not None:
        if val_size >= 1 and val_size < 0:
            raise 'Make sure `val_size` is in range [0,1)'

        n_val = math.floor(val_size * len(x))
        n_train = len(x) - n_val

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=n_train, test_size=n_val, random_state=seed)

    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)
    print('x_val.shape', x_val.shape)
    print('y_val.shape', y_val.shape)
    print('x_test.shape', x_test.shape)
    print('y_test.shape', y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test
