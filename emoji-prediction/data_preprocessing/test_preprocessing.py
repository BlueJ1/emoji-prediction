import pandas as pd
from pathlib import Path

from parse_to_df import parse_to_df

data_path = Path(__file__).parent.parent / 'data'
emoji_path = data_path / 'emojis.txt'
file_name = 'train.txt'
df = parse_to_df(data_path=data_path, size_to_read=5 * 1024 ** 2)

with open(emoji_path, 'r') as f:
    emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

# count occurrences of emojis
emoji_counts = df['sequence_emojis'].apply(pd.Series).stack().value_counts()
print(emoji_counts)
