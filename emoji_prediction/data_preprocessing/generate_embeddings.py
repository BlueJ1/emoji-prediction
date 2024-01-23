import pandas as pd


def embed_words(df, index_to_word, word_to_embedding):
    """
    This function embeds words in a dataframe using index_to_word
    and word_to_embedding dictionaries. It takes a dataframe with a 'word' column holding word indexes,
    a dictionary mapping index to word, and a dictionary mapping word to embedding. It returns a dataframe 
    with all embeddings as columns.

    Parameters:
    df (DataFrame): The DataFrame with a 'word' column holding word indexes.
    index_to_word (dict): The dictionary mapping index to word.
    word_to_embedding (dict): The dictionary mapping word to embedding.

    Returns:
    DataFrame: The DataFrame with all embeddings as columns.
    """
    # Map the word indexes to actual words
    df['word'] = df['word'].map(index_to_word)
    # Map the words to their embeddings
    df['embedding'] = df['word'].map(word_to_embedding)

    # Remove all rows with NaN embeddings
    df = df.dropna(subset=['embedding'])
    # Print the number of remaining rows
    print(str(len(df)) + " rows left after dropping nan values")

    # Split the embeddings into separate columns named 'embedding_0', 'embedding_1', etc.
    df = df.join(pd.DataFrame(df['embedding'].to_list(), index=df.index))

    # Drop the original 'word' and 'embedding' columns
    df.drop(['word'], axis=1, inplace=True)
    df.drop(['embedding'], axis=1, inplace=True)

    return df
