import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.four_gram_class import FourGram
from models.evaluate_predictions import evaluate_predictions


def four_gram_data(df):
    X = df['words'].apply(lambda x: FourGram(x)).values
    y = df['emoji'].apply(int).values
    return X, y


def four_gram(i, X_train, y_train, X_test, y_test, results_dict, _):
    data_path = Path(__file__).parent.parent / 'data'
    emoji_path = data_path / 'emojis.txt'

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    unique_4_grams = set(X_train)
    print(f'{len(unique_4_grams)} unique 4-grams in data')

    unique_emojis, counts = np.unique(y_train, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    four_gram_dict = {}
    for words, emoji in tqdm(zip(X_train, y_train)):
        if words in four_gram_dict:
            four_gram_dict[words][emoji] += 1
        else:
            four_gram_dict[words] = np.zeros(len(emoji_vocab))
            four_gram_dict[words][emoji] += 1

    # select argmax for each row
    for key, value in tqdm(four_gram_dict.items()):
        four_gram_dict[key] = np.argmax(value)

    predictions = []
    for words in X_test:
        if words not in four_gram_dict:
            predictions.append(most_common_emoji)
        else:
            predictions.append(four_gram_dict[words])

    results_dict[i] = evaluate_predictions(predictions, y_test)
