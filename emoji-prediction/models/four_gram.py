import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import pandas as pd

try:
    from models.four_gram_class import FourGram
    from models.evaluate_predictions import evaluate_predictions
except ModuleNotFoundError:
    from four_gram_class import FourGram
    from evaluate_predictions import evaluate_predictions


def four_gram_data(df):
    X = df['words'].apply(lambda x: FourGram(x)).values
    y = df['emoji'].apply(int).values
    return X, y


def four_gram(i, X_train, y_train, X_test, y_test, results_dict, _):
    data_path = Path(__file__).parent.parent / 'data'
    emoji_path = data_path / 'emojis.txt'

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    unique_4_grams = set(X_train)
    print(f'{len(unique_4_grams)} unique 4-grams in data')

    unique_emojis, counts = np.unique(y_train, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    four_gram_dict = {}
    for words, emoji in tqdm(zip(X_train, y_train)):
        if words in four_gram_dict:
            four_gram_dict[words][emoji] += 1
        else:
            four_gram_dict[words] = np.zeros(len(emoji_vocab))
            four_gram_dict[words][emoji] += 1

    # select argmax for each row
    for key, value in tqdm(four_gram_dict.items()):
        four_gram_dict[key] = np.argmax(value)

    predictions = []
    for words in X_test:
        if words not in four_gram_dict:
            predictions.append(most_common_emoji)
        else:
            predictions.append(four_gram_dict[words])

    results_dict[i] = evaluate_predictions(predictions, y_test)


def generate_four_gram(X, y, _):
    data_path = Path(__file__).parent.parent / 'data'
    emoji_path = data_path / 'emojis.txt'

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    unique_4_grams = set(X)
    print(f'{len(unique_4_grams)} unique 4-grams in data')

    unique_emojis, counts = np.unique(y, return_counts=True)
    most_common_emoji = unique_emojis[np.argmax(counts)]

    four_gram_dict = {}
    for words, emoji in tqdm(zip(X, y), total=len(X)):
        if words in four_gram_dict:
            four_gram_dict[words][emoji] += 1
        else:
            four_gram_dict[words] = np.zeros(len(emoji_vocab))
            four_gram_dict[words][emoji] += 1

    # select argmax for each row
    for key, value in tqdm(four_gram_dict.items()):
        four_gram_dict[key] = np.argmax(value)

    pickle.dump((four_gram_dict, most_common_emoji), open('four_gram.pkl', 'wb+'))


def four_gram_process_api_data(sentence: str, index: int, word_vocab: dict) -> FourGram:
    words = sentence.lower().split()
    if index < 2:
        words_before = [''] * (2 - index) + words[:index]
    else:
        words_before = words[index - 2:index]

    if index > len(words) - 2:
        words_after = words[index + 1:] + [''] * (index + 2 - len(words))
    else:
        words_after = words[index:index + 2]

    words_around = words_before + words_after
    words_around = [word_vocab[word] if word in word_vocab else word_vocab[''] for word in words_around]
    four_gram = FourGram(np.array(words_around))

    return four_gram


def four_gram_api_predict(sentence: str, index: int):
    data_path = Path(__file__).parent.parent / 'data'
    vocab_path = data_path / 'vocab.txt'
    emoji_path = data_path / 'emojis.txt'
    model_path = Path(__file__).parent.parent / 'models'
    four_gram_model_path = model_path / 'four_gram.pkl'

    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {idx: emoji[:-1] for idx, emoji in enumerate(f.readlines())}

    four_gram_dict, most_common_emoji = pickle.load(open(four_gram_model_path, 'rb'))

    four_gram: FourGram = four_gram_process_api_data(sentence, index, word_vocab)

    if four_gram in four_gram_dict:
        predicted_idx = four_gram_dict[four_gram]
    else:
        predicted_idx = most_common_emoji

    return emoji_vocab[predicted_idx]


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'data'
    data_file = 'words_around_emoji_index.pkl'
    df = pd.read_pickle(data_path / data_file)

    X, y = four_gram_data(df)

    generate_four_gram(X, y, None)

    print(four_gram_api_predict('hate you', 2))
