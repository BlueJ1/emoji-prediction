import keras.src.backend
import numpy as np
import pandas as pd
from multiprocessing import Process, Manager
from sklearn.model_selection import StratifiedKFold
from pickle import dump
from pathlib import Path
import tensorflow as tf
import sys

from balance_dataset import balance_multiclass_dataset


def main(run_id, parameters):
    # k-fold cross validation
    k = 5
    results = {}
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

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

                    # print("Loaded data, fold", i)

                    if parameter_dict['balance_dataset']:
                        X_train, y_train = balance_multiclass_dataset(X_train, y_train)
                        # print("Balanced data, fold", i)
                    parameter_dict['hyperparameters']['gpu_id'] = i % num_gpus if num_gpus > 0 else -1

                    p = Process(target=parameter_dict["evaluate"], args=(
                        i, X_train, y_train, X_test, y_test, results_dict, parameter_dict['hyperparameters']))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                results_dict = dict(results_dict)
        else:
            results_dict = {}
            for i, (train_index, test_index) in enumerate(cv.split(np.zeros(X.shape[0]), np.zeros(y.shape[0]))):
                if isinstance(X, np.ndarray):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                else:
                    X_train, X_test = tf.gather(X, indices=train_index), tf.gather(X, indices=test_index)
                    y_train, y_test = tf.gather(y, indices=train_index), tf.gather(y, indices=test_index)

                # print("Loaded data, fold", i)

                if parameter_dict['balance_dataset']:
                    X_train, y_train = balance_multiclass_dataset(X_train, y_train)
                    # print("Balanced data, fold", i)

                parameter_dict['hyperparameters']['gpu_id'] = i % num_gpus if num_gpus > 0 else -1
                parameter_dict['evaluate'](i, X_train, y_train, X_test, y_test, results_dict,
                                           parameter_dict['hyperparameters'])
                # print("Evaluated model on data, fold", i)

                keras.backend.clear_session()

        for val in results_dict.values():
            results[parameter_dict['name']].append(val)
        print(f"Weighted F1 score of {parameter_dict['name']}: "
              f"{np.average([results_dict[i]['weighted_f1_score'] for i in range(k)])}")

    # save results
    with open(f'results_{run_id}.pkl', 'wb') as f:
        dump(results, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Missing argument. Please specify which hyperparameter set to evaluate.")
        print("Usage: python cross_validation.py (classic_ml|nn) [run_id]")
        print("Example: python cross_validation.py classic_ml test_run")
        exit(1)
    if sys.argv[1] == 'classic_ml':
        from classic_ml_hyperparam_configs import parameters
    elif sys.argv[1] == 'nn':
        from nn_hyperparam_configs import parameters
    else:
        print("Invalid argument", sys.argv[1])
        print("Please specify which hyperparameter set to evaluate.")
        print("Usage: python cross_validation.py [classic_ml|nn] [run_id]")
        print("Example: python cross_validation.py classic_ml test_run")
        exit(1)

    run_id = sys.argv[2] if len(sys.argv) > 2 else 'test'

    main(run_id, parameters)
