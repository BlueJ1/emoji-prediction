import numpy as np
import pandas as pd
from pathlib import Path
from parse_to_df import parse_to_df


def keep_words_surrounding_emoji(row, num_of_words_before, num_of_words_after, index_to_word, word_to_embedding,
                                 embedding_shape, want_embeddings=False):
    # Get indices of nonzero elements in the second column

    padded_emojis = np.pad(row['sequence_emojis'], (num_of_words_before - 1, num_of_words_after), 'constant',
                           constant_values=(0, 0))
    if want_embeddings:
        padded_words = np.pad(row['sequence_words'], (num_of_words_before - 1, num_of_words_after), 'constant',
                              constant_values=(0, 0))
        padded_words = np.array(
            [word_to_embedding.get(index_to_word[i], np.zeros(embedding_shape)) for i in padded_words])

    else:
        padded_words = np.pad(row['sequence_words'], (num_of_words_before - 1, num_of_words_after), 'constant',
                              constant_values=(0, 0))
    nonzero_indices = np.nonzero(padded_emojis)[0]
    # Create a sublist from the first column based on nonzero indices

    new_column1 = [[padded_words[j] for j in range(i - num_of_words_before + 1, i + num_of_words_after + 1)] for i in
                   nonzero_indices]

    # Update the second column to keep only nonzero elements
    new_column2 = [padded_emojis[i] for i in nonzero_indices]

    return pd.Series({'words': new_column1, 'emoji': new_column2})


def generate_dataframes(size_in_MB):
    data_path = Path(__file__).parent.parent / 'data'
    file_name = 'train.txt'
    vocab_path = data_path / 'vocab.txt'
    embedding_path = data_path / 'glove.6B.50d.txt'

    df = parse_to_df(data_path=data_path, file_path=data_path / file_name, size_to_read=int(size_in_MB * 1024 ** 2))
    df.to_csv(data_path / 'train.csv')

    flattened_df = pd.DataFrame(
        {'word': df['sequence_words'].apply(pd.Series).stack(),
         'emoji': df['sequence_emojis'].apply(pd.Series).stack()})
    df_word_before_emoji = flattened_df[flattened_df['emoji'] != 0]

    df_word_before_emoji.to_pickle(data_path / f'word_before_emoji_index.pkl')

    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
        index_to_word = {float(i): w for w, i in word_vocab.items()}
        # vectorized_ix_to_word = np.vectorize(index_to_word.get)

    with open(embedding_path, 'r', encoding='utf-8') as f:
        word_to_embedding = {}
        for line in f.readlines():
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype=float)
            word_to_embedding[word] = embedding

        # For some reason this function doesn't work on vectors of words
        # vectorized_word_to_glove = np.vectorize(lambda x: word_to_embedding.get(x, np.zeros(embedding.shape).tolist()))

    new_df = df.apply(lambda x: keep_words_surrounding_emoji(x, 2, 2, None, None, None, False), axis=1)
    df_words_around_emoji = new_df[new_df.astype(bool).any(axis=1)].apply(
        lambda col: col.explode() if col.dtype == 'O' else col)
    df_words_around_emoji.to_pickle(data_path / f'words_around_emoji_index.pkl')

    embedded_df = df.apply(
        lambda x: keep_words_surrounding_emoji(x, 2, 2, index_to_word, word_to_embedding, embedding.shape, True),
        axis=1).apply(
        lambda col: col.explode() if col.dtype == 'O' else col)
    embedded_df = embedded_df.dropna(subset=['words'])

    sum_first_col = lambda row: pd.Series({'words': np.sum(row['words'], axis=0), 'emoji': row['emoji']})
    concatenate_first_col = lambda row: pd.Series(
        {'words': np.concatenate(row['words'], axis=0), 'emoji': row['emoji']})

    sum_of_embeddings_df = pd.get_dummies(embedded_df.apply(sum_first_col, axis=1), columns=['emoji'],
                                          prefix=['Target_class'])
    concatenation_of_embedding_df = pd.get_dummies(embedded_df.apply(concatenate_first_col, axis=1), columns=['emoji'],
                                                   prefix=['Target_class'])

    sum_of_embeddings_df.to_pickle(data_path / f'word_around_emoji_sum_of_embeddings.pkl')
    concatenation_of_embedding_df.to_pickle(data_path /
                                            f'word_around_emoji_concatenation_of_embeddings.pkl')


generate_dataframes(5)
