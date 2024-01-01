import numpy as np
import pandas as pd
from multiprocessing import Process, Manager
from sklearn.model_selection import StratifiedKFold
from pickle import dump
from pathlib import Path
import tensorflow as tf

from models.four_gram import four_gram, four_gram_data
from models.one_gram import one_gram, one_gram_data
from models.baseline import baseline, baseline_data
from models.mlp_unified import mlp_data, train_fold

parameters = [
    # dict(
    #     name='baseline',
    #     data_preprocessing=baseline_data,
    #     data_file='word_before_emoji_index.pkl',
    #     evaluate=baseline,
    #     hyperparameters=dict(),
    #     mlp=False,
    #     parallel=True
    # ),
    # dict(
    #     name='one_gram',
    #     data_preprocessing=one_gram_data,
    #     data_file='word_before_emoji_index.pkl',
    #     evaluate=one_gram,
    #     hyperparameters=dict(),
    #     mlp=False,
    #     parallel=True
    # ),
    # dict(
    #     name='four_gram',
    #     data_preprocessing=four_gram_data,
    #     data_file='words_around_emoji_index.pkl',
    #     evaluate=four_gram,
    #     hyperparameters=dict(),
    #     mlp=False,
    #     parallel=True
    # ),
    dict(
        name='mlp_concat',
        data_preprocessing=mlp_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_fold,
        hyperparameters=dict(input_dim=200,
                             output_dim=49,
                             lr=1e-5,
                             num_epochs=1000,
                             batch_size=2048,
                             gpu_id=0),
        mlp=True,
        parallel=True
    ),
    dict(
        name='mlp_sum',
        data_preprocessing=mlp_data,
        data_file='word_around_emoji_concatenation_of_embeddings.pkl',
        evaluate=train_fold,
        hyperparameters=dict(input_dim=50,
                             output_dim=49,
                             lr=1e-5,
                             num_epochs=1000,
                             batch_size=2048,
                             gpu_id=0),
        mlp=True,
        parallel=True
    )
]


if __name__ == '__main__':
    # k-fold cross validation
    k = 5
    results = {}
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))

    for parameter_dict in parameters:
        print(f'Running {parameter_dict["name"]} model')
        data_dir = Path(__file__).parent / 'data'
        data_file = parameter_dict['data_file']
        df = pd.read_pickle(data_dir / data_file)

        X, y = parameter_dict['data_preprocessing'](df)

        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        results[parameter_dict['name']] = []

        if parameter_dict['parallel']:
            with Manager() as manager:
                results_dict = manager.dict()
                processes = []
                for i, (train_index, test_index) in enumerate(cv.split(np.zeros(X.shape[0]), np.zeros(y.shape[0]))):
                    if isinstance(X, np.ndarray):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                    else:
                        X_train, X_test = tf.gather(X, indices=train_index), tf.gather(X, indices=test_index)
                        y_train, y_test = tf.gather(y, indices=train_index), tf.gather(y, indices=test_index)
                    parameter_dict['hyperparameters']['gpu_id'] = i % num_gpus if num_gpus > 0 else -1

                    p = Process(target=parameter_dict["evaluate"], args=(
                    i, X_train, y_train, X_test, y_test, results_dict, parameter_dict['hyperparameters']))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                results_dict = dict(results_dict)
        else:
            for i, (train_index, test_index) in enumerate(cv.split(np.zeros(X.shape[0]), y)):
                results_dict = {}
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                parameter_dict['hyperparameters']['gpu_id'] = i % num_gpus
                parameter_dict['evaluate'](i, X_train, y_train, X_test, y_test, results_dict,
                                           parameter_dict['hyperparameters'])

        for val in results_dict.values():
            results[parameter_dict['name']].append(val)

    # save results
    with open('results.pkl', 'wb') as f:
        dump(results, f)
