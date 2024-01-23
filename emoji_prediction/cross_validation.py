import numpy as np
import pandas as pd
from multiprocessing import Process, Manager
from sklearn.model_selection import StratifiedKFold
from pickle import dump
from pathlib import Path
import sys

# import local module
from balance_dataset import balance_multiclass_dataset


def main(run_id, parameters):
    """
    Main function to perform k-fold cross validation on the given parameters.

    Args:
    run_id (str): The run identifier.
    parameters (list): List of dictionaries containing model parameters.

    Returns:
    None
    """

    # define the number of folds for cross validation
    k = 5
    results = {}

    # iterate over each parameter dictionary
    for parameter_dict in parameters:
        print(f'Running {parameter_dict["name"]} model')

        # load the data
        data_dir = Path(__file__).parent / 'data'
        data_file = parameter_dict['data_file']
        df = pd.read_pickle(data_dir / data_file)

        # preprocess the data
        X, y = parameter_dict['data_preprocessing'](df)

        # initialize the cross validation
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        results[parameter_dict['name']] = []

        # check if parallel processing is required
        if parameter_dict['parallel']:
            with Manager() as manager:
                results_dict = manager.dict()
                processes = []
                for i, (train_index, test_index) in enumerate(cv.split(np.zeros(X.shape[0]), np.zeros(y.shape[0]))):
                    # split the data into training and testing sets
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # balance the dataset if required
                    if parameter_dict['balance_dataset']:
                        X_train, y_train = balance_multiclass_dataset(X_train, y_train)

                    # start a new process for evaluation
                    p = Process(target=parameter_dict["evaluate"], args=(
                        i, X_train, y_train, X_test, y_test, results_dict, parameter_dict['hyperparameters']))
                    p.start()
                    processes.append(p)

                # wait for all processes to finish
                for p in processes:
                    p.join()

                results_dict = dict(results_dict)
        else:
            results_dict = {}
            for i, (train_index, test_index) in enumerate(cv.split(np.zeros(X.shape[0]), np.zeros(y.shape[0]))):
                # split the data into training and testing sets
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # balance the dataset if required
                if parameter_dict['balance_dataset']:
                    X_train, y_train = balance_multiclass_dataset(X_train, y_train)

                # evaluate the model
                parameter_dict['evaluate'](i, X_train, y_train, X_test, y_test, results_dict,
                                           parameter_dict['hyperparameters'])

        # calculate the average weighted F1 score
        for val in results_dict.values():
            results[parameter_dict['name']].append(val)
        print(f"Weighted F1 score of {parameter_dict['name']}: "
              f"{np.average([results_dict[i]['weighted_f1_score'] for i in range(k)])}")

        # save the results
        with open(f'results_{run_id}.pkl', 'wb') as f:
            dump(results, f)


if __name__ == '__main__':
    """
    Entry point of the script. It checks the command line arguments and calls the main function.
    """

    # check the command line arguments
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

    # get the run id
    run_id = sys.argv[2] if len(sys.argv) > 2 else 'test'

    # call the main function
    main(run_id, parameters)
