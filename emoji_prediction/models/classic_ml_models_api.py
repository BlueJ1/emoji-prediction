import pickle
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
FOLD_NUMBER = 5


def getdata():
    data_path = Path(__file__).parent.parent / 'data'
    file_name = 'word_around_emoji_sum_of_embeddings.pkl'

    df = pd.read_pickle(data_path / file_name)
    # print(f"These are the columns of our {file_name} pandas: {df.columns}")
    # print(f"This is the number of rows of our {file_name} pandas: {len(df)}")
    # print(f"Max length of words: {df['words'].apply(len).max()}")

    expanded_df = df.apply(lambda row: pd.Series(row['words']), axis=1)

    # print(f"These are the columns of our expanded_df pandas"
    #       f"These are only the words : {expanded_df.columns}")

    X = expanded_df.values  # the words, now as indexes
    y = df.iloc[:, 1:].values  # the target classes (emojis, 50 classes)
    y = np.argmax(y, axis=1)  # we take the index of the max value, which is the emoji
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    # print(f"X_train shape: {X_train.shape}")
    X_train = scaler.fit_transform(X_train)
    print(f"X_train shape after fit_transform: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    X_test = scaler.transform(X_test)
    print(f"X_test shape after transform: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def add_emoji(prediction, index, text):
    # get emoji from emoji dict
    data_path = Path(__file__).parent.parent / 'data'
    file_name = 'emojis.txt'
    with open(data_path / file_name, 'r', encoding='utf-8') as f:
        emoji_dict = {idx: emoji[:-1] for idx, emoji in enumerate(f.readlines())}

    print(f"The prediction number is {prediction},")
    # print(f"and the emoji is {emoji_dict[prediction]}")
    # prediction = emoji_dict[prediction]
    # add prediction to text in the index position
    return text[:index] + prediction + text[index:]


def embed_text(text):
    data_path = Path(__file__).parent.parent / 'data'
    file_name = 'glove.6B.50d.txt'
    with open(data_path / file_name, 'r', encoding='utf-8') as f:
        glove = {}
        for line in f.readlines():
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype=float)
            glove[word] = embedding

    # get the embeddings of the words in the text
    text = text.split()
    text = [glove[word] for word in text]
    text = np.sum(text, axis=0)
    print(f"Text embedded: {text}")
    return text


def logreq_api(text: str, index: int, X_train, y_train, X_test) -> str:
    model = train("log_reg", X_train, y_train, X_test)
    if model == "":
        print("You need to train the model first")
        return
    text = embed_text(text)
    # make a 2d array
    text = np.array([text])
    print(f"Text shape: {text.shape}")
    prediction = np.argmax(model.predict(text))
    response = add_emoji(prediction, index, text)
    print(response)
    pass


def train(model_name: str, X_train, y_train, X_test):
    model = ""
    if model_name == "log_reg":
        print("Chose Logistic Regression")
        model = LogisticRegression(C=1, penalty="elasticnet", l1_ratio=0.5, solver="saga",
                                   max_iter=300, n_jobs=-1, verbose=1)
    else:
        print("You need to choose a model")
        return
    # print first row of X_train
    print(f"This is an example of input {X_train[0]}, shape: {X_train[0].shape}")
    print("Training the model")
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test)}")
    # save in a pickle file
    print("Saving the model")
    os.makedirs("models", exist_ok=True)

    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = getdata()
    log_reg_api = logreq_api("i love you", 2, X_train, y_train, X_test)
    #train("log_reg", X_train, y_train, X_test)
