import pandas as pd


def embed_words(df, index_to_word, word_to_embedding):
    """
    Embeds words in a dataframe using index_to_word
    and word_to_embedding dictionaries.
    :param df: dataframe with a 'word' column holding word indexes
    :param index_to_word: dictionary mapping index to word
    :param word_to_embedding: dictionary mapping word to embedding
    :return: dataframe with all embeddings as columns
    """

    df['word'] = df['word'].map(index_to_word)
    df['embedding'] = df['word'].map(word_to_embedding)

    # take out all embeddings that are nan
    df = df.dropna(subset=['embedding'])
    # print how much we have left
    print(str(len(df)) + " rows left after dropping nan values")

    # put each of them in a column named embedding_0, embedding_1, etc.
    df = df.join(pd.DataFrame(df['embedding'].to_list(), index=df.index))

    df.drop(['word'], axis=1, inplace=True)
    df.drop(['embedding'], axis=1, inplace=True)
    return df
