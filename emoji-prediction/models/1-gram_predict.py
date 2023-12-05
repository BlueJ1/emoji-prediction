# this file is used to generate a 1-gram language model for emoji prediction
# it is generated from the word_before_emoji_index file

import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path(__file__).parent.parent / 'data'
file_name = 'word_before_emoji_index.pkl'
vocab_path = data_path / 'vocab.txt'
emoji_path = data_path / 'emojis.txt'

with open(vocab_path, 'r', encoding='utf-8') as f:
    word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

with open(emoji_path, 'r', encoding='utf-8') as f:
    emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

df = pd.read_pickle(data_path / file_name)

one_gram_matrix = np.zeros((len(word_vocab), len(emoji_vocab)))

for index, row in df.iterrows():
    for word, emoji in zip(row['word'], row['emoji']):
        one_gram_matrix[word, emoji] += 1

one_gram_matrix = one_gram_matrix / one_gram_matrix.sum(axis=0)

np.save(data_path / 'one_gram_matrix.npy', one_gram_matrix)
