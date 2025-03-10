import numpy as np
import pandas as pd
from pathlib import Path
from parse_to_df import parse_to_df
from time import time


# This function keeps the words surrounding an emoji in a given row. It pads the sequence of emojis and words with
# zeros to ensure that the number of words before and after the emoji is maintained. It also converts the words to
# their corresponding embeddings.
def keep_words_surrounding_emoji(row, num_of_words_before, num_of_words_after,
                                 index_to_word, word_to_embedding: dict,
                                 embedding_shape):
    # Padding the sequence of emojis and words
    padded_emojis = np.pad(row['sequence_emojis'], (
        num_of_words_before - 1, num_of_words_after), 'constant',
                           constant_values=(0, 0))
    padded_words = np.pad(row['sequence_words'], (
        num_of_words_before - 1, num_of_words_after),
                          'constant', constant_values=(0, 0))
    # Converting words to their corresponding embeddings
    padded_words_emb = np.array(
        [word_to_embedding.get(
            index_to_word[i],
            np.zeros(embedding_shape)) for i in padded_words])

    nonzero_indices = np.nonzero(padded_emojis)[0]
    new_column1 = [padded_words[
                   i - num_of_words_before + 1:i + num_of_words_after + 1
                   ] for i in nonzero_indices]
    new_column1_emb = [padded_words_emb[
                       i - num_of_words_before + 1:i + num_of_words_after + 1
                       ] for i in nonzero_indices]
    new_column2 = padded_emojis[nonzero_indices]

    return new_column1, new_column1_emb, new_column2


# This function sums the first column of a given row.
def sum_first_col(row):
    return pd.Series(
        {'words': np.sum(row['words'], axis=0),
         'emoji': row['emoji']})


# This function concatenates the first column of a given row.
def concatenate_first_col(row):
    return pd.Series(
        {'words': np.concatenate(row['words'], axis=0),
         'emoji': row['emoji']})


# This function generates dictionaries for word to index, index to word, and word to embedding mappings.
def generate_dictionaries(vocab_file="vocab.txt"):
    data_path = Path(__file__).parent.parent / 'data'
    vocab_path = data_path / vocab_file
    embedding_path = data_path / 'glove.6B.50d.txt'

    # Generating word to index and index to word mappings
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
        index_to_word = {float(i): w for w, i in word_vocab.items()}

    # Generating word to embedding mapping
    with open(embedding_path, 'r', encoding='utf-8') as f:
        word_to_embedding = {}
        for line in f.readlines():
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype=float)
            word_to_embedding[word] = embedding

    return index_to_word, word_to_embedding, embedding.shape


# This function loads a basic dataframe from a given file.
def load_basic_dataframe(size_in_MB, vocab_file="vocab.txt"):
    data_path = Path(__file__).parent.parent / 'data'
    return parse_to_df(data_path=data_path, vocab_file=vocab_file,
                       size_to_read=int(size_in_MB * 1024 ** 2))


# This function generates a training dataframe from a given dataframe.
def generate_train_dataframe(
        df, word_to_embedding, index_to_word, embedding_shape):
    data_path = Path(__file__).parent.parent / 'data'

    # Flattening the dataframe
    flattened_df = pd.DataFrame(
        {'word': df['sequence_words'].apply(pd.Series).stack(),
         'emoji': df['sequence_emojis'].apply(pd.Series).stack()})

    df_word_before_emoji = flattened_df[flattened_df['emoji'] != 0]
    df_word_before_emoji.to_pickle(data_path / 'word_before_emoji_index.pkl')

    words_around_emoji_data_ix = []
    words_around_emoji_data_emb = []
    for _, row in df.iterrows():
        new_col1_ix, new_col1_emb, new_col2 = keep_words_surrounding_emoji(
            row, num_of_words_before=2, num_of_words_after=2,
            index_to_word=index_to_word, word_to_embedding=word_to_embedding,
            embedding_shape=embedding_shape)
        words_around_emoji_data_ix.extend(zip(new_col1_ix, new_col2))
        words_around_emoji_data_emb.extend(zip(new_col1_emb, new_col2))

    df_words_around_emoji = pd.DataFrame(
        words_around_emoji_data_ix, columns=['words', 'emoji'])
    df_words_around_emoji.to_pickle(data_path / 'words_around_emoji_index.pkl')

    embedded_df = pd.DataFrame(
        words_around_emoji_data_emb, columns=['words', 'emoji'])

    sum_of_embeddings_df = pd.get_dummies(
        embedded_df.apply(sum_first_col, axis=1),
        columns=['emoji'], prefix=['Target_class'])
    sum_of_embeddings_df.to_pickle(
        data_path / 'word_around_emoji_sum_of_embeddings.pkl')

    concatenation_of_embedding_df = pd.get_dummies(
        embedded_df.apply(concatenate_first_col, axis=1),
        columns=['emoji'], prefix=['Target_class'])
    concatenation_of_embedding_df.to_pickle(
        data_path / 'word_around_emoji_concatenation_of_embeddings.pkl')


# Main function
if __name__ == '__main__':
    t = time()
    df = load_basic_dataframe(100, vocab_file='vocab.txt')
    ix_to_word, word_to_glove, shape = generate_dictionaries()
    generate_train_dataframe(df, word_to_glove, ix_to_word, shape)
    print(f'Time taken: {time() - t}')
