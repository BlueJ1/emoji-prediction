import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from parse_to_df import parse_to_df

data_path = Path(__file__).parent.parent / 'data'
emoji_path = data_path / 'emojis.txt'
file_name = 'train.txt'
df = parse_to_df(data_path=data_path, size_to_read=500 * 1024 ** 2)

with open(emoji_path, 'r') as f:
    emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
    index_to_emoji = {float(i): w for w, i in emoji_vocab.items()}

# count occurrences of top 20 emojis
emoji_counts = df['sequence_emojis'].apply(pd.Series).stack().value_counts()
emoji_counts.index = emoji_counts.index.map(index_to_emoji)

# plot
plt.figure(figsize=(20, 10))
plt.bar(emoji_counts.index[1:], emoji_counts.values[1:])
plt.xticks(rotation=90)
plt.xlabel('Emoji')
plt.ylabel('Count')
plt.title('Emoji counts')
plt.show()

print(emoji_counts)
