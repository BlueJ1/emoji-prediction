# this file is used to generate a 4-gram language model for emoji prediction
# the given words are the two words before and after the emoji
# it is generated from the words_around_emoji_index.pkl file

import numpy as np
from pathlib import Path
from baseline import baseline_predict


def four_gram_predict(wordss):
    data_path = Path(__file__).parent.parent / 'data'
    dict_file = data_path / 'four_gram_dict.npy'

    four_gram_dict = np.load(dict_file)

    predictions = []
    for words in wordss:
        if words not in four_gram_dict:
            predictions.append(baseline_predict(words[1])[0])
        predictions.append(four_gram_dict[words])
    return predictions
