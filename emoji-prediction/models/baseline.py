# this file is used to predict emojis using the baseline model of guessing the most common emoji

import numpy as np

from models.evaluate_predictions import evaluate_predictions


def baseline_data(df):
    X = df['word'].values
    y = df['emoji'].values
    return X, y


def baseline(X_train, y_train, X_test, y_test, results, _):
    unique_emojis, counts = np.unique(y_train, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    predictions = np.repeat(most_common_emoji, len(X_test))

    results.append(evaluate_predictions(predictions, y_test))
