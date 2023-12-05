# this file is used to generate a 4-gram language model for emoji prediction
# the given words are the two words before and after the emoji
# it is generated from the words_around_emoji_index.pkl file

import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path(__file__).parent.parent / 'data'
file_name = 'words_around_emoji_index.pkl'
vocab_path = data_path / 'vocab.txt'
