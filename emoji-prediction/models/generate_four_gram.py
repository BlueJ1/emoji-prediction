# this file is used to generate a 1-gram language model for emoji prediction
# it is generated from the words_around_emoji_index.pkl file

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from four_gram_class import FourGram


data_path = Path(__file__).parent.parent / 'data'
model_path = Path(__file__).parent.parent / 'models'
file_name = 'words_around_emoji_index.pkl'
vocab_path = data_path / 'vocab.txt'
emoji_path = data_path / 'emojis.txt'

with open(emoji_path, 'r', encoding='utf-8') as f:
    emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

df = pd.read_pickle(data_path / file_name)
print(df.head())

print(df.info())

# unique 4-grams in data
unique_4_grams = set(df['words'].apply(lambda x: FourGram(x)))
print(f'{len(unique_4_grams)} unique 4-grams in data')

four_gram_dict = {}
for index, row in tqdm(df.iterrows()):
    words, emoji = FourGram(row['words']), row['emoji']
    if words in four_gram_dict:
        four_gram_dict[words][emoji] += 1
    else:
        four_gram_dict[words] = np.zeros(len(emoji_vocab))
        four_gram_dict[words][emoji] += 1

# select argmax for each row
for key, value in tqdm(four_gram_dict.items()):
    four_gram_dict[key] = np.argmax(value)


# transform dict to matrix
four_gram_matrix = np.array(list(four_gram_dict.items()), dtype=object)

# save matrix
np.save(model_path / 'four_gram_dict.npy', four_gram_matrix)

# print(four_gram_matrix.shape)
# print(four_gram_matrix[:10])

# load matrix
loaded_matrix = np.load(model_path / 'four_gram_dict.npy', allow_pickle=True)
print(loaded_matrix.shape)
print(type(loaded_matrix[0][0]))
