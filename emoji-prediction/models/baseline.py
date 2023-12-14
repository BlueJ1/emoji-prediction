# this file is used to predict emojis using the baseline model of guessing the most common emoji

import numpy as np


def baseline_data(df):
    X = df['word'].values
    y = df['emoji'].values
    return X, y


def baseline(X_train, y_train, X_test, y_test, results, _):
    unique_emojis, counts = np.unique(y_train, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    predictions = np.repeat(most_common_emoji, len(X_test))

    results.append(np.mean(predictions == y_test))
