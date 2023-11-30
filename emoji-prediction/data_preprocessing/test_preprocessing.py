import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from parse_to_df import parse_to_df
from generate_embeddings import embed_words
from dim_reduction import pca, t_sne

matplotlib.use('TkAgg')
# Use the TkAgg backend (replace with appropriate backend for your system)


data_path = Path(__file__).parent.parent / 'data'
file_name = 'train.txt'
emoji_path = data_path / 'emojis.txt'
vocab_path = data_path / 'vocab.txt'
embedding_path = Path(__file__).parent.parent.parent / 'data/glove.6B.50d.txt'

df = parse_to_df(data_path=data_path, size_to_read=5 * 1024 ** 2)
print(df.dtypes)
all_equal_length = True
for i, row in df.iterrows():
    if len(row['sequence_words']) != len(row['sequence_emojis']):
        all_equal_length = False
        print(f'Row {i} has unequal length')
        print(row['sequence_words'])
        print(row['sequence_emojis'])

df.to_csv(data_path / 'train.csv')


def keep_words_before_emoji(row):
    # Get indices of nonzero elements in the second column
    # code with less than 50 characters
    nonzero_indices = []
    for i, val in enumerate(row['sequence_emojis']):
        if val != 0:
            nonzero_indices.append(i)

    # Create a sublist from the first column based on nonzero indices
    new_column1 = [row['sequence_words'][i] for i in nonzero_indices]

    # Update the second column to keep only nonzero elements
    new_column2 = [val for val in row['sequence_emojis'] if val != 0]

    return pd.Series({'word': new_column1, 'emoji': new_column2})


new_df = df.apply(keep_words_before_emoji, axis=1)
# Remove rows with empty lists
new_df_filtered = new_df[new_df.astype(bool).any(axis=1)]
df_words_before_emoji = new_df_filtered.apply(
    # Flatten the lists to individual entries
    lambda col: col.explode() if col.dtype == 'O' else col)

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

# plot emoji freq
plt.figure(figsize=(20, 10))
plt.bar(emoji_counts.index[1:], emoji_counts.values[1:])
plt.xticks(rotation=90)
plt.xlabel('Emoji')
plt.ylabel('Count')
plt.title('Emoji counts')
plt.savefig('emoji_counts.png')

# plot word freq
plt.figure(figsize=(20, 10))
plt.bar(word_counts.index[:50], word_counts.values[:50])
plt.xticks(rotation=90)
plt.xlabel('Word')
plt.ylabel('Count')
plt.title('Word before emoji count')
plt.savefig('word_counts.png')

top_emojis = emoji_counts.index[1:1 + 5]

new_list_of_emojis = []
for top in top_emojis:
    new_list_of_emojis.append(emoji_vocab[top])

filtered = df_words_before_emoji['emoji'].isin(new_list_of_emojis)

top_filtered_df = df_words_before_emoji[filtered].reset_index(drop=True)

with open(embedding_path, 'r', encoding='utf-8') as f:
    word_to_embedding = {}
    for lines in f.readlines():
        lines = lines.split()
        word_to_embedding.update({lines[0]: lines[1:]})
    # the first element is the word, the rest are 50 values of the embedding

embedded_df = embed_words(top_filtered_df, index_to_word, word_to_embedding)

# save to csv
embedded_df.to_csv(data_path / 'embedded.csv')
print(embedded_df.head())
pca_df = pca(embedded_df)
t_sne_df = t_sne(embedded_df)
