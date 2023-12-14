import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pickle import dump
from pathlib import Path

from models.four_gram import four_gram, four_gram_data
from models.one_gram import one_gram, one_gram_data
from models.baseline import baseline, baseline_data

parameters = [
    dict(
        name='baseline',
        data_preprocessing=baseline_data,
        data_file='word_before_emoji_index.pkl',
        evaluate=baseline,
        hyperparameters=dict()
    ),
    dict(
        name='one_gram',
        data_preprocessing=one_gram_data,
        data_file='word_before_emoji_index.pkl',
        evaluate=one_gram,
        hyperparameters=dict()
    ),
    dict(
        name='four_gram',
        data_preprocessing=four_gram_data,
        data_file='words_around_emoji_index.pkl',
        evaluate=four_gram,
        hyperparameters=dict()
    )]

# k-fold cross validation
k = 5
results = {}

for parameter_dict in parameters:
    print(f'Running {parameter_dict["name"]} model')
    data_dir = Path(__file__).parent / 'data'
    data_file = parameter_dict['data_file']
    df = pd.read_pickle(data_dir / data_file)

    X, y = parameter_dict['data_preprocessing'](df)
    print(f'y shape: {y.shape}')
    print(y.dtype)

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results[parameter_dict['name']] = []

    for i, (train_index, test_index) in enumerate(cv.split(np.zeros(X.shape[0]), y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        parameter_dict['evaluate'](X_train, y_train, X_test, y_test, results[parameter_dict['name']],
                                   parameter_dict['hyperparameters'])

# save results
with open('results.pkl', 'wb') as f:
    dump(results, f)
