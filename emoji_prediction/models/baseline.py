import numpy as np
from models.evaluate_predictions import evaluate_predictions


def baseline_data(df):
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
    y = df['emoji'].values
    return X, y


def baseline(i, X_train, y_train, X_test, y_test, results_dict, _):
    """
    Trains a baseline model and evaluates its performance

    Parameters
    ----------
        i : int
            index of the current fold in cross-validation
        X_train : array
            training data - not used
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
    unique_emojis, counts = np.unique(y_train, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    predictions = np.repeat(most_common_emoji, len(X_test))

    results_dict[i] = evaluate_predictions(predictions, y_test)
