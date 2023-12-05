# this file is used to generate a 1-gram language model for emoji prediction
# it is generated from the words_around_emoji_index.pkl file

from pathlib import Path
import pandas as pd

data_path = Path(__file__).parent.parent / 'data'
model_path = Path(__file__).parent.parent / 'models'
file_name = 'words_around_emoji_index.pkl'
vocab_path = data_path / 'vocab.txt'
emoji_path = data_path / 'emojis.txt'

with open(vocab_path, 'r', encoding='utf-8') as f:
    word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

with open(emoji_path, 'r', encoding='utf-8') as f:
    emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

df = pd.read_pickle(data_path / file_name)
print(df.head())

# unique 4-grams in data
unique_4_grams = set(df['words'].apply(lambda x: str(x)))
print(len(unique_4_grams))
print(list(unique_4_grams)[:10])
print(len(df["words"]))

exit()

four_gram_matrix = np.zeros((len(word_vocab), len(word_vocab), len(word_vocab), len(word_vocab), len(emoji_vocab)))

for index, row in df.iterrows():
    ...

one_gram_matrix = np.zeros((len(word_vocab), len(emoji_vocab)))

for index, row in df.iterrows():
    word, emoji = row['word'], row['emoji']
    one_gram_matrix[word, emoji] += 1

# normalize the matrix, while accounting for zero rows
one_gram_matrix = one_gram_matrix / np.maximum(one_gram_matrix.sum(axis=1, keepdims=True), 1)

np.save(model_path / 'one_gram_matrix.npy', one_gram_matrix)
