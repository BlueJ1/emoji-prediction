import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.evaluate_predictions import evaluate_predictions


def basic_ml_data(df):
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
    expanded_df = df.apply(lambda row: pd.Series(row['words']), axis=1)
    X = expanded_df.values
    y = df.iloc[:, 1:].values
    y = np.argmax(y, axis=1)

    return X, y


def train_rf(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    """
    Trains a Random Forest model and evaluates its performance

    Parameters
    ----------
        fold_number : int
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
        hyperparameters : dict
            dictionary containing hyperparameters for the model
    """
    n_estimators = hyperparameters['n_estimators']
    criterion = hyperparameters['criterion']
    max_features = hyperparameters['max_features']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_svm(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    """
    Trains a Support Vector Machine model and evaluates its performance

    Parameters
    ----------
        fold_number : int
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
        hyperparameters : dict
            dictionary containing hyperparameters for the model
    """
    kernel = hyperparameters['kernel']
    C = hyperparameters['C']
    tol = hyperparameters['tol']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel=kernel, C=C, tol=tol, max_iter=30000, cache_size=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_qda(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    """
    Trains a Quadratic Discriminant Analysis model and evaluates its performance

    Parameters
    ----------
        fold_number : int
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
        hyperparameters : dict
            dictionary containing hyperparameters for the model
    """
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_k_nbh(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    """
    Trains a K-Nearest Neighbors model and evaluates its performance

    Parameters
    ----------
        fold_number : int
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
        hyperparameters : dict
            dictionary containing hyperparameters for the model
    """
    num_neighbors = hyperparameters['num_neighbors']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=num_neighbors, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_naive_bayes(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    """
    Trains a Naive Bayes model and evaluates its performance

    Parameters
    ----------
        fold_number : int
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
        hyperparameters : dict
            dictionary containing hyperparameters for the model
    """
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_log_reg(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    """
    Trains a Logistic Regression model and evaluates its performance

    Parameters
    ----------
        fold_number : int
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
        hyperparameters : dict
            dictionary containing hyperparameters for the model
    """
    C = hyperparameters['C']
    penalty = hyperparameters['penalty']
    l1_ratio = hyperparameters['l1_ratio']
    solver = hyperparameters['solver']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(C=C, penalty=penalty, l1_ratio=l1_ratio, solver=solver, max_iter=20000, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def nb_data(df):
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
    X = np.stack(df['words'].values)
    y = df['emoji'].values
    return X, y
