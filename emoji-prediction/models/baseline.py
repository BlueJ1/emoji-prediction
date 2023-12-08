# this file is used to predict emojis using the baseline model

import numpy as np


def baseline_predict(words):
    random_baseline = np.load('random_baseline.npy')
    predictions = [random_baseline] * len(words)
    return predictions
