import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.evaluate_predictions import evaluate_predictions


def one_gram_data(df):
    X = df['word'].values
    y = df['emoji'].apply(int).values
    return X, y


def one_gram(X_train, y_train, X_test, y_test, results, _):
    data_path = Path(__file__).parent.parent / 'data'
    emoji_path = data_path / 'emojis.txt'

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    # create one gram model
    unique_emojis, counts = np.unique(y_train, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    one_gram_dict = {}

    for word, emoji in tqdm(zip(X_train, y_train)):
        if word in one_gram_dict:
            one_gram_dict[word][emoji] += 1
        else:
            one_gram_dict[word] = np.zeros(len(emoji_vocab))
            one_gram_dict[word][emoji] += 1

    # make predictions
    predictions = []
    for word in X_test:
        if word not in one_gram_dict:
            predictions.append(most_common_emoji)
        else:
            predictions.append(np.argmax(one_gram_dict[word]))

    results.append(evaluate_predictions(predictions, y_test))
