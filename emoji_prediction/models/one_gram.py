import pickle

import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from models.evaluate_predictions import evaluate_predictions
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.join(
        os.path.join(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..'), 'emoji_prediction'), 'models'))
    from evaluate_predictions import evaluate_predictions


def one_gram_data(df):
    """
    Transforms the dataframe into X and y arrays for model training

    Parameters
    ----------
        df : DataFrame
            a dataframe containing 'word' and 'emoji' columns

    Returns
    -------
        tuple
            X and y arrays for model training
    """
    X = df['word'].values
    y = df['emoji'].apply(int).values
    return X, y


def one_gram(i, X_train, y_train, X_test, y_test, results_dict, _):
    """
    Trains a one-gram model and evaluates its performance

    Parameters
    ----------
        i : int
            index of the current fold in cross-validation
        X_train : array
            training data
        y_train : array
            training labels
        X_test : array
            testing data
        y_test : array
            testing labels
        results_dict : dict
            dictionary to store the results
        _ : None
            placeholder for unused parameter
    """
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
    pickle.dump((one_gram_dict, most_common_emoji), open('one_gram.pkl', 'wb+'))
    # make predictions
    predictions = []
    for word in X_test:
        if word not in one_gram_dict:
            predictions.append(most_common_emoji)
        else:
            predictions.append(np.argmax(one_gram_dict[word]))

    results_dict[i] = evaluate_predictions(predictions, y_test)


def one_gram_process_api_data(sentence: str, index: int):
    """
    Processes a sentence into a one-gram for API prediction

    Parameters
    ----------
        sentence : str
            input sentence
        index : int
            index of the word to predict the emoji for

    Returns
    -------
        None
    """
    data_path = Path(__file__).parent.parent / 'data'
    emoji_path = data_path / 'emojis.txt'
    word_path = data_path / 'words.txt'

    with(open(emoji_path, 'r', encoding='utf-8')) as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    with(open(word_path, 'r', encoding='utf-8')) as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}
