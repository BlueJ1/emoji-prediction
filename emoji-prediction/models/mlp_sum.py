from multiprocessing import Process, Manager
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.metrics import F1Score
from keras.optimizers.legacy import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import numpy as np


def setup_gpu(gpu_id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        except RuntimeError as e:
            print(e)


class TqdmMetricsProgressBarCallback(Callback):
    def __init__(self, total_epochs, validation_data=None, eval_interval=10):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = None
        self.validation_data = validation_data
        self.eval_interval = eval_interval
        self.val_loss = 0
        self.val_acc = 0
        self.val_f1_score = 0

    def on_train_begin(self, logs=None):
        self.progress_bar = tqdm(total=self.total_epochs, unit='epoch', bar_format='{l_bar}{bar}{r_bar}')

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('accuracy', 0) * 100  # Convert to percentage
        f1_score = logs.get('f1_score', 0)  # F1 score

        if self.validation_data and (epoch + 1) % self.eval_interval == 0:
            self.val_loss, self.val_acc, self.val_f1_score = (
                self.model.evaluate(self.validation_data[0], self.validation_data[1], verbose=0, batch_size=4096))
        self.progress_bar.set_postfix_str(f"Loss: {loss:.2f}, Accuracy: {acc:.2f}%, F1 Score: {f1_score:.2f}, "
                                          f"Val Loss: {self.val_loss:.2f}, Val Accuracy: {self.val_acc:.2f}%, "
                                          f"Val F1 Score: {self.val_f1_score:.2f}")
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()


def train_fold(fold_number, fold_data, results_dict):
    gpu_id = fold_data['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    X = fold_data['X']
    y = fold_data['y']
    train_idxs = fold_data['train_idxs']
    test_idxs = fold_data['test_idxs']
    num_epochs = fold_data['num_epochs']

    # Build the MLP model
    model = keras.Sequential([
        layers.Input(shape=(50,)),  # Assuming 50 features
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),  # Adding dropout for regularization
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),  # Adding dropout for regularization
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),  # Adding dropout for regularization
        layers.Dense(128, activation='relu'),  # fewer units
        layers.Dropout(0.1),  # Adding dropout for regularization
        layers.Dense(49, activation='softmax')  # Assuming 49 emoji classes
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='CategoricalCrossentropy',
        metrics=[F1Score(average="weighted", dtype=tf.float32), "accuracy"])

    # Standardize the features
    scaler = StandardScaler()
    # print(train_idxs.shape)
    X_train = scaler.fit_transform(tf.gather(X, train_idxs))
    y_train = tf.gather(y, train_idxs)
    X_test = scaler.transform(tf.gather(X, test_idxs))
    y_test = tf.gather(y, test_idxs)

    # Train the model (in silent mode, verbose=0)
    tqdm_callback = TqdmMetricsProgressBarCallback(num_epochs, validation_data=(X_test, y_test),
                                                   eval_interval=10)
    history = model.fit(X_train, y_train,
                        epochs=num_epochs, batch_size=128, verbose=0,
                        callbacks=[tqdm_callback])
    # Evaluate the model on the validation data
    evaluation = model.evaluate(
        X_test, y_test, verbose=1, batch_size=4096)

    results_dict[fold_number] = evaluation


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'data'
    model_path = Path(__file__).parent.parent / 'models'
    file_name = 'word_around_emoji_sum_of_embeddings.pkl'

    df = pd.read_pickle(data_path / file_name)

    # Expand the words embedding column
    def expand_array(row):
        return pd.Series(row['words'])

    expanded_df = df.apply(expand_array, axis=1)

    # Concatenate the expanded columns with the original DataFrame
    result_df = pd.concat([df, expanded_df], axis=1)
    # Drop the original array_column
    result_df = result_df.drop('words', axis=1)
    # print(result_df.head())

    X = result_df.iloc[:, 49:].values
    y = result_df.iloc[:, :49].values

    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)

    # k-fold cross validation
    k = 5
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    num_epochs = 1000
    all_scores = []

    parallel = True

    if parallel:
        with Manager() as manager:
            results_dict = manager.dict()
            processes = []
            num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))

            for i, (train_idxs, test_idxs) in enumerate(kfold.split(X, np.argmax(y, axis=1))):
                fold_data = {
                    'gpu_id': i % num_gpus if num_gpus > 0 else -1,
                    'train_idxs': train_idxs,
                    'test_idxs': test_idxs,
                    'X': X,
                    'y': y,
                    'num_epochs': num_epochs
                }
                p = Process(target=train_fold, args=(i, fold_data, results_dict))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            results_dict = dict(results_dict)
    else:
        for i, (train_idxs, test_idxs) in enumerate(kfold.split(X, np.argmax(y, axis=1))):
            results_dict = {}
            fold_data = {
                'gpu_id': i % len(tf.config.experimental.list_physical_devices('GPU')),  # assuming num_gpus is the
                # number of GPUs you have
                'train_idxs': train_idxs,
                'test_idxs': test_idxs,
                'X': X,
                'y': y,
                'num_epochs': num_epochs
            }
            train_fold(i, fold_data, results_dict)

    for val in results_dict.values():
        all_scores.append(val)

    all_scores = np.array(all_scores)
    print(np.average(all_scores[:, :4], axis=0))
