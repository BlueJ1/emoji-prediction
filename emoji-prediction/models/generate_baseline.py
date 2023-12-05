# the random baseline always predicts the most frequent emoji in the training set

import pandas as pd
import numpy as np

from pathlib import Path

data_path = Path(__file__).parent.parent / 'data'
model_path = Path(__file__).parent.parent / 'models'
file_name = 'word_before_emoji_index.pkl'

df = pd.read_pickle(data_path / file_name)

emoji_counts = df['emoji'].value_counts()

random_baseline = int(emoji_counts.index[0])
print(emoji_counts)
print("------------")
print(random_baseline)

# save the random baseline
np.save(model_path / 'random_baseline.npy', random_baseline)
