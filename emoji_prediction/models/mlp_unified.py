import keras
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.metrics import F1Score
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
from pathlib import Path
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
        acc = logs.get('accuracy', 0)  # Accuracy
        f1_score = logs.get('f1_score', 0)  # F1 score

        if self.validation_data and (epoch + 1) % self.eval_interval == 0:
            self.val_loss, self.val_acc, self.val_f1_score = (
                self.model.evaluate(self.validation_data[0], self.validation_data[1], verbose=0, batch_size=1024))
        self.progress_bar.set_postfix_str(f"Loss: {loss:.2f}, Accuracy: {acc * 100:.2f}%, "
                                          f"F1 Score: {f1_score * 100:.2f}%, "
                                          f"Val Loss: {self.val_loss:.2f}, Val Accuracy: {self.val_acc * 100:.2f}%, "
                                          f"Val F1 Score: {self.val_f1_score * 100:.2f}%")
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()


def train_fold(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    gpu_id = hyperparameters['gpu_id']
    if gpu_id >= 0:
        setup_gpu(gpu_id)

    input_dim = hyperparameters['input_dim']
    output_dim = 49
    learning_rate = hyperparameters['lr']
    num_epochs = hyperparameters['num_epochs']
    batch_size = hyperparameters['batch_size']

    # Build the MLP model
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
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
        run_eagerly=True,
        metrics=[F1Score(average="weighted", dtype=tf.float32), "accuracy"])

    # Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model (in silent mode, verbose=0)
    tqdm_callback = TqdmMetricsProgressBarCallback(num_epochs, validation_data=(X_test, y_test),
                                                   eval_interval=1)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    history = model.fit(X_train, y_train,
                        epochs=num_epochs, batch_size=batch_size, verbose=0,
                        callbacks=[tqdm_callback])
    # Evaluate the model on the validation data
    evaluation = model.evaluate(
        X_test, y_test, verbose=1, batch_size=batch_size)

    session.close()

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
    # print(X)
    # print(y)
    return X, y


def train_and_save_mlp(X, y, hyperparameters):
    data_path = Path(__file__).parent.parent / 'data'
    emoji_path = data_path / 'emojis.txt'

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

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
        optimizer=SGD(learning_rate=learning_rate),
        loss='CategoricalCrossentropy',
        metrics=[F1Score(average="weighted", dtype=tf.float32), "accuracy"])

    # Standardize the features
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # Train the model (in silent mode, verbose=0)
    tqdm_callback = TqdmMetricsProgressBarCallback(num_epochs, eval_interval=np.inf)
    model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=[tqdm_callback])

    model.save(Path(__file__).parent.parent / 'models' / 'mlp_concat.keras')


def mlp_concat_process_api_data(sentence: str, index: int, word_vocab: dict, word_to_embedding: dict) -> tf.Tensor:
    words = sentence.lower().split()
    if index < 2:
        words_before = [''] * (2 - index) + words[:index]
    else:
        words_before = words[index - 2:index]

    if index > len(words) - 2:
        words_after = words[index:] + [''] * (index + 2 - len(words))
    else:
        words_after = words[index:index + 2]

    words_around = words_before + words_after
    words_around = [word_vocab[word] if word in word_vocab else word_vocab[''] for word in words_around]
    embeddings_around = [word_to_embedding[word] if word in word_to_embedding else np.zeros(50) for word in
                         words_around]
    embeddings_around = np.concatenate(embeddings_around, axis=0)
    embeddings_around = tf.cast([embeddings_around,], tf.float32)

    return embeddings_around


def mlp_concat_api_predict(sentence: str, index: int):
    data_path = Path(__file__).parent.parent / 'data'
    vocab_path = data_path / 'vocab.txt'
    emoji_path = data_path / 'emojis.txt'
    emoji_to_unicode_path = data_path / 'emoji_name_to_unicode.txt'
    model_path = Path(__file__).parent.parent / 'models'
    mlp_model_path = model_path / 'mlp_concat.keras'

    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {idx: emoji[:-1] for idx, emoji in enumerate(f.readlines())}

    with open(emoji_to_unicode_path, 'r', encoding='utf-8') as f:
        emoji_to_unicode = {emoji: unicode for emoji, unicode in [line.split() for line in f.readlines()]}

    model = keras.models.load_model(mlp_model_path)

    embedding_path = data_path / 'glove.6B.50d.txt'
    with open(embedding_path, 'r', encoding='utf-8') as f:
        word_to_embedding = {}
        for line in f.readlines():
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype=float)
            word_to_embedding[word] = embedding

    embeddings_around = mlp_concat_process_api_data(sentence, index, word_vocab, word_to_embedding)
    print(embeddings_around.shape)

    predicted_idx = np.argmax(model.predict(embeddings_around)[0])

    return emoji_to_unicode[emoji_vocab[predicted_idx]]


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'data'
    data_file = 'word_around_emoji_concatenation_of_embeddings.pkl'
    df = pd.read_pickle(data_path / data_file)

    X, y = mlp_data(df)

    train_and_save_mlp(X, y, dict(input_dim=200,
                                  output_dim=49,
                                  lr=1e-4,
                                  num_epochs=20,
                                  batch_size=1024,
                                  gpu_id=0))

    print(mlp_concat_api_predict('love you', 2))
