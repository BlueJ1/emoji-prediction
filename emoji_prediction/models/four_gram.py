import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
try:
    from emoji_prediction.models.evaluate_predictions import evaluate_predictions
except ModuleNotFoundError:
    import sys
    from evaluate_predictions import evaluate_predictions


class FourGram:
    """
    A class used to represent a FourGram

    ...

    Attributes
    ----------
    words : list
        a list of words that make up the four-gram

    Methods
    -------
    __getitem__(self, item):
        Returns the word at the given index in the four-gram

    __len__(self):
        Returns the length of the four-gram

    __str__(self):
        Returns a string representation of the four-gram

    __repr__(self):
        Returns a string representation of the four-gram

    __eq__(self, other):
        Checks if the four-gram is equal to another four-gram

    __hash__(self):
        Returns a hash value of the four-gram
    """

    def __init__(self, words):
        """
        Constructs all the necessary attributes for the FourGram object.

        Parameters
        ----------
            words : list
                a list of words that make up the four-gram
        """
        self.words = words

    def __getitem__(self, item):
        """
        Returns the word at the given index in the four-gram

        Parameters
        ----------
            item : int
                index of the word in the four-gram

        Returns
        -------
            str
                word at the given index in the four-gram
        """
        return self.words[item]

    def __len__(self):
        """
        Returns the length of the four-gram

        Returns
        -------
            int
                length of the four-gram
        """
        return len(self.words)

    def __str__(self):
        """
        Returns a string representation of the four-gram

        Returns
        -------
            str
                string representation of the four-gram
        """
        return str(self.words)

    def __repr__(self):
        """
        Returns a string representation of the four-gram

        Returns
        -------
            str
                string representation of the four-gram
        """
        return str(self.words)

    def __eq__(self, other):
        """
        Checks if the four-gram is equal to another four-gram

        Parameters
        ----------
            other : FourGram
                another four-gram to compare with

        Returns
        -------
            bool
                True if the four-gram is equal to the other four-gram, False otherwise
        """
        return np.array_equal(np.array(self.words), np.array(other.words))

    def __hash__(self):
        """
        Returns a hash value of the four-gram

        Returns
        -------
            int
                hash value of the four-gram
        """
        return hash(str(np.array(self.words)))


def four_gram_data(df):
    """
    Transforms the dataframe into X and y arrays for model training

    Parameters
    ----------
        df : DataFrame
            a dataframe containing 'words' and 'emoji' columns

    Returns
    -------
        tuple
            X and y arrays for model training
    """
    X = df['words'].apply(lambda x: FourGram(x)).values
    y = df['emoji'].apply(int).values
    return X, y


def four_gram(i, X_train, y_train, X_test, y_test, results_dict, _):
    """
    Trains a four-gram model and evaluates its performance

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

    unique_emojis, counts = np.unique(y_train, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    four_gram_dict = {}
    for words, emoji in zip(X_train, y_train):
        if words in four_gram_dict:
            four_gram_dict[words][emoji] += 1
        else:
            four_gram_dict[words] = np.zeros(len(emoji_vocab))
            four_gram_dict[words][emoji] += 1

    # select argmax for each row
    for key, value in four_gram_dict.items():
        four_gram_dict[key] = np.argmax(value)

    predictions = []
    for words in X_test:
        if words not in four_gram_dict:
            predictions.append(most_common_emoji)
        else:
            predictions.append(four_gram_dict[words])

    results_dict[i] = evaluate_predictions(predictions, y_test)


def generate_four_gram(X, y, _):
    data_path = Path(__file__).parent.parent / 'data'
    emoji_path = data_path / 'emojis.txt'

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    unique_4_grams = set(X)
    print(f'{len(unique_4_grams)} unique 4-grams in data')

    unique_emojis, counts = np.unique(y, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    four_gram_dict = {}
    for words, emoji in tqdm(zip(X, y), total=len(X)):
        if words in four_gram_dict:
            four_gram_dict[words][emoji] += 1
        else:
            four_gram_dict[words] = np.zeros(len(emoji_vocab))
            four_gram_dict[words][emoji] += 1

    # select argmax for each row
    for key, value in tqdm(four_gram_dict.items()):
        four_gram_dict[key] = np.argmax(value)

    pickle.dump((four_gram_dict, most_common_emoji), open('four_gram.pkl', 'wb+'))


def four_gram_process_api_data(sentence: str, index: int, word_vocab: dict) -> FourGram:
    words = sentence.lower().split()
    if index < 2:
        words_before = [''] * (2 - index) + words[:index]
    else:
        words_before = words[index - 2:index]

    if index > len(words) - 2:
        words_after = words[index + 1:] + [''] * (index + 2 - len(words))
    else:
        words_after = words[index:index + 2]

    words_around = words_before + words_after
    words_around = [word_vocab[word] if word in word_vocab else word_vocab[''] for word in words_around]
    four_gram = FourGram(np.array(words_around))

    return four_gram


def four_gram_api_predict(sentence: str, index: int):
    data_path = Path(__file__).parent.parent / 'data'
    vocab_path = data_path / 'vocab.txt'
    emoji_path = data_path / 'emojis.txt'
    model_path = Path(__file__).parent.parent / 'models'
    four_gram_model_path = model_path / 'four_gram.pkl'

    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {idx: emoji[:-1] for idx, emoji in enumerate(f.readlines())}

    four_gram_dict, most_common_emoji = pickle.load(open(four_gram_model_path, 'rb'))

    four_gram: FourGram = four_gram_process_api_data(sentence, index, word_vocab)

    if four_gram in four_gram_dict:
        predicted_idx = four_gram_dict[four_gram]
    else:
        predicted_idx = most_common_emoji

    return emoji_vocab[predicted_idx]


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'data'
    data_file = 'words_around_emoji_index.pkl'
    df = pd.read_pickle(data_path / data_file)

    X, y = four_gram_data(df)

    generate_four_gram(X, y, None)

    print(four_gram_api_predict('love you', 2))
