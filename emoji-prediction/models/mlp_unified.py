import keras
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.metrics import F1Score
from keras.optimizers.legacy import Adam, SGD
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm


def setup_gpu(gpu_id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        except RuntimeError as e:
            print(e)


class TqdmMetricsProgressBarCallback(Callback):
    def __init__(self, total_epochs, validation_data=None, eval_interval=1):
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


def train_fold(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    input_dim = hyperparameters['input_dim']
    output_dim = hyperparameters['output_dim']
    learning_rate = hyperparameters['lr']
    num_epochs = hyperparameters['num_epochs']
    batch_size = hyperparameters['batch_size']

    # Build the MLP model
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),  # Assuming 50 features
        layers.Dense(512, activation='relu'),
        # layers.Dropout(0.2),  # Adding dropout for regularization
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.15),  # Adding dropout for regularization
        layers.Dense(128, activation='relu'),
        # layers.Dropout(0.1),  # Adding dropout for regularization
        layers.Dense(128, activation='relu'),  # fewer units
        # layers.Dropout(0.05),  # Adding dropout for regularization
        layers.Dense(output_dim, activation='softmax')  # Assuming 49 emoji classes
    ])

    # Compile the model
    model.compile(
        optimizer=SGD(learning_rate=learning_rate),  # Adam(learning_rate=learning_rate),
        loss='CategoricalCrossentropy',
        metrics=[F1Score(average="weighted", dtype=tf.float32), "accuracy"])

    # Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model (in silent mode, verbose=0)
    tqdm_callback = TqdmMetricsProgressBarCallback(num_epochs, validation_data=(X_test, y_test),
                                                   eval_interval=1)
    history = model.fit(X_train, y_train,
                        epochs=num_epochs, batch_size=batch_size, verbose=0,
                        callbacks=[tqdm_callback])
    # Evaluate the model on the validation data
    evaluation = model.evaluate(
        X_test, y_test, verbose=1, batch_size=2 * batch_size)

    results_dict[fold_number] = evaluation


def mlp_data(df):
    expanded_df = df.apply(lambda row: pd.Series(row['words']), axis=1)

    # Concatenate the expanded columns with the original DataFrame
    result_df = pd.concat([df, expanded_df], axis=1)

    # Drop the original array_column
    result_df = result_df.drop('words', axis=1)

    X = result_df.iloc[:, 49:].values
    y = result_df.iloc[:, :49].values

    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    print(X.shape, y.shape)
    print(X)
    print(y)
    return X, y
