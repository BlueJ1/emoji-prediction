import pandas as pd
from mlp_sum import data_path

file_name = 'word_around_emoji_concatenation_of_embeddings.pkl'

df = pd.read_pickle(data_path / file_name)
# print column names
print(df.columns)
# print max length of words
print(df['words'].apply(len).max())
