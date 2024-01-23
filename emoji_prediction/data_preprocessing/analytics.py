import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from parse_to_df import parse_to_df
from generate_embeddings import embed_words
from dim_reduction import pca, t_sne
matplotlib.use('TkAgg')

# Define the path to the data directory
data_path = Path(__file__).parent.parent / 'data'
# Define the name of the file to be parsed
file_name = 'train.txt'
# Define the path to the emoji vocabulary file
emoji_path = data_path / 'emojis.txt'
# Define the path to the word vocabulary file
vocab_path = data_path / 'vocab.txt'
# Define the path to the word embeddings file
embedding_path = Path(__file__).parent.parent.parent / 'data/glove.6B.50d.txt'

# Parse the file into a DataFrame
df = parse_to_df(data_path=data_path, size_to_read=5 * 1024 ** 2)
# Print the data types of the DataFrame columns
print(df.dtypes)

# Check if all sequences have equal length
all_equal_length = True
for i, row in df.iterrows():
    if len(row['sequence_words']) != len(row['sequence_emojis']):
        all_equal_length = False
        print(f'Row {i} has unequal length')
        print(row['sequence_words'])
        print(row['sequence_emojis'])

# Save the DataFrame to a CSV file
df.to_csv(data_path / 'train.csv')


def keep_words_before_emoji(row):
    """
    This function keeps the words that appear before an emoji in a given row.
    It updates the first column to keep only the words that appear before an emoji.
    It also updates the second column to keep only the emojis that have a word before them.

    Parameters: row (Series): A row from a DataFrame. The first column should contain a list of words and the second
    column should contain a list of emojis.

    Returns: Series: A Series where the first column contains a list of words that appear before an emoji and the
    second column contains a list of emojis that have a word before them.
    """
    # Get indices of nonzero elements in the second column
    nonzero_indices = []
    for i, val in enumerate(row['sequence_emojis']):
        if val != 0:
            nonzero_indices.append(i)

    # Create a sublist from the first column based on nonzero indices
    new_column1 = [row['sequence_words'][i] for i in nonzero_indices]

    # Update the second column to keep only nonzero elements
    new_column2 = [val for val in row['sequence_emojis'] if val != 0]

    return pd.Series({'word': new_column1, 'emoji': new_column2})


# Apply the function to each row of the DataFrame
new_df = df.apply(keep_words_before_emoji, axis=1)
# Remove rows with empty lists
new_df_filtered = new_df[new_df.astype(bool).any(axis=1)]
# Flatten the lists to individual entries
df_words_before_emoji = new_df_filtered.apply(
    lambda col: col.explode() if col.dtype == 'O' else col)

# Load the emoji vocabulary
with open(emoji_path, 'r') as f:
    emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
    index_to_emoji = {float(i): w for w, i in emoji_vocab.items()}

# Count occurrences of emojis
emoji_counts = df['sequence_emojis'].apply(pd.Series).stack().value_counts()
# Map the indices to the actual emojis
emoji_counts.index = emoji_counts.index.map(index_to_emoji)

# Load the word vocabulary
with open(vocab_path, 'r', encoding='utf-8') as f:
    word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
    index_to_word = {float(i): w for w, i in word_vocab.items()}

# Count occurrences of words
word_counts = df_words_before_emoji['word'].value_counts()
# Map the indices to the actual words
word_counts.index = word_counts.index.map(index_to_word)

# Plot the frequency of each emoji
plt.figure(figsize=(20, 10))
plt.bar(emoji_counts.index[1:], emoji_counts.values[1:])
plt.xticks(rotation=90)
plt.xlabel('Emoji')
plt.ylabel('Count')
plt.title('Emoji counts')
plt.savefig('emoji_counts.png')

# Plot the frequency of each word
plt.figure(figsize=(20, 10))
plt.bar(word_counts.index[:50], word_counts.values[:50])
plt.xticks(rotation=90)
plt.xlabel('Word')
plt.ylabel('Count')
plt.title('Word before emoji count')
plt.savefig('word_counts.png')

# Get the top 5 most frequent emojis
top_emojis = emoji_counts.index[1:1 + 5]

# Map the top emojis to their indices
new_list_of_emojis = []
for top in top_emojis:
    new_list_of_emojis.append(emoji_vocab[top])

# Filter the DataFrame to keep only the rows with the top emojis
filtered = df_words_before_emoji['emoji'].isin(new_list_of_emojis)
top_filtered_df = df_words_before_emoji[filtered].reset_index(drop=True)

# Load the word embeddings
with open(embedding_path, 'r', encoding='utf-8') as f:
    word_to_embedding = {}
    for lines in f.readlines():
        lines = lines.split()
        word_to_embedding.update({lines[0]: lines[1:]})

# Embed the words in the DataFrame
embedded_df = embed_words(top_filtered_df, index_to_word, word_to_embedding)

# Save the DataFrame to a CSV file
embedded_df.to_csv(data_path / 'embedded.csv')
print(embedded_df.head())
# Apply PCA to the DataFrame
pca_df = pca(embedded_df)
# Apply t-SNE to the DataFrame
t_sne_df = t_sne(embedded_df)
