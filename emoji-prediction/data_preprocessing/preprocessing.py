import numpy as np
import pandas as pd
from pathlib import Path
from parse_to_df import parse_to_df

data_path = Path(__file__).parent.parent / 'data'
file_name = 'train.txt'
emoji_path = data_path / 'emojis.txt'
vocab_path = data_path / 'vocab.txt'
df = parse_to_df(data_path=data_path, size_to_read=int(0.25 * 1024 ** 2))
print(df.dtypes)

all_equal_length = True
for i, row in df.iterrows():
    if len(row['sequence_words']) != len(row['sequence_emojis']):
        all_equal_length = False
        print(f'Row {i} has unequal length')
        print(row['sequence_words'])
        print(row['sequence_emojis'])

df.to_csv(data_path / 'train.csv')

flattend_df = pd.DataFrame(
    {'word': df['sequence_words'].apply(pd.Series).stack(), 'emoji': df['sequence_emojis'].apply(pd.Series).stack()})
df_word_before_emoji = flattend_df[flattend_df['emoji'] != 0]


def keep_words_surrounding_emoji(row, num_of_words_before, num_of_words_after):
    # Get indices of nonzero elements in the second column

    padded_emojis = np.pad(row['sequence_emojis'], (num_of_words_before - 1, num_of_words_after), 'constant',
                           constant_values=(0, 0))
    padded_words = np.pad(row['sequence_words'], (num_of_words_before - 1, num_of_words_after), 'constant',
                          constant_values=(0, 0))

    nonzero_indices = np.nonzero(padded_emojis)[0]
    # Create a sublist from the first column based on nonzero indices

    new_column1 = [[padded_words[j] for j in range(i - num_of_words_before + 1, i + num_of_words_after + 1)] for i in
                   nonzero_indices]

    # Update the second column to keep only nonzero elements
    new_column2 = [padded_emojis[i] for i in nonzero_indices]

    return pd.Series({'words': new_column1, 'emoji': new_column2})


new_df = df.apply(lambda x: keep_words_surrounding_emoji(x, 2, 2), axis=1)
new_df_filtered = new_df[new_df.astype(bool).any(axis=1)].apply(
    lambda col: col.explode() if col.dtype == 'O' else col)

print(new_df_filtered)
print(df_word_before_emoji)

with open(emoji_path, 'r') as f:
    emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
    index_to_emoji = {float(i): w for w, i in emoji_vocab.items()}

# count occurrences of emojis
emoji_counts = df['sequence_emojis'].apply(pd.Series).stack().value_counts()
emoji_counts.index = emoji_counts.index.map(index_to_emoji)

with open(vocab_path, 'r', encoding='utf-8') as f:
    word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
    index_to_word = {float(i): w for w, i in word_vocab.items()}

# count occurrences of words
word_counts = df_words_before_emoji['word'].value_counts()
word_counts.index = word_counts.index.map(index_to_word)

# print(word_counts)
# print(emoji_counts)
