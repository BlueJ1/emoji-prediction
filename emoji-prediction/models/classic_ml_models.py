from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.evaluate_predictions import evaluate_predictions


def basic_ml_data(df):
    expanded_df = df.apply(lambda row: pd.Series(row['words']), axis=1)

    # Drop the original array_column
    df = df.drop('words', axis=1)

    # Concatenate the expanded columns with the original DataFrame
    result_df = pd.concat([df, expanded_df], axis=1)

    X = result_df.iloc[:, 49:].values
    y = result_df.iloc[:, :49].values
    y = np.argmax(y, axis=1)

    # print(X.shape, y.shape)
    # print(X)
    # print(y)
    return X, y


def setup_gpu(gpu_id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        except RuntimeError as e:
            print(e)


def train_rf(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    n_estimators = hyperparameters['n_estimators']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_svm(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    tol = hyperparameters['tol']
    C = hyperparameters['C']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=C, tol=tol)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_gaussian_process(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    n_restarts = hyperparameters['n_restarts']
    max_iter_pred = hyperparameters['max_iter_pred']
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GaussianProcessClassifier(n_restarts_optimizer=n_restarts, max_iter_predict=max_iter_pred, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_k_nbh(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

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
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def train_log_reg(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def nb_data(df):
    X = np.stack(df['words'].values)
    y = df['emoji'].values
    return X, y


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    file_name = 'word_around_emoji_sum_of_embeddings.pkl'

    df = pd.read_pickle(data_path / file_name)
    # print column names
    # print(df.columns)
    # print max length of words
    X, y = basic_ml_data(df)
    # X, y = nb_data(df)
    # print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_y_pred = gnb.predict(X_test)
    gnb_accuracy = accuracy_score(y_test, gnb_y_pred)
    gnb_rf_f1 = f1_score(y_test, gnb_y_pred, average='weighted')
    classification_rep = classification_report(y_test, gnb_y_pred)
    print("Naive Bayes:")
    print(f"Accuracy: {gnb_accuracy}")
    print("F1 Score: ", gnb_rf_f1)
    print("Classification Report:\n", classification_rep)

    del gnb

    # # SVM with RBF Kernel
    # svm_classifier = SVC(kernel='rbf')
    # svm_classifier.fit(X_train, y_train)
    # svm_y_pred = svm_classifier.predict(X_test)
    # svm_accuracy = accuracy_score(y_test, svm_y_pred)
    # svm_classification_rep = classification_report(y_test, svm_y_pred)
    # print("\nSVM with RBF Kernel:")
    # print(f"Accuracy: {svm_accuracy}")
    # print("Classification Report:\n", svm_classification_rep)

    # Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    rf_y_pred = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    rf_f1 = f1_score(y_test, rf_y_pred, average='weighted')
    rf_classification_rep = classification_report(y_test, rf_y_pred)
    print("Random Forest:")
    print(f"Accuracy: {rf_accuracy}")
    print(f"F1 Score: {rf_f1}")
    print("Classification Report:\n", rf_classification_rep)
