from pathlib import Path

import numpy as np
import pandas as pd


def parse_to_df(data_path: Path = None, file_path: Path = None, vocab_file="vocab.txt",
                size_to_read: int = 0):
    """
    This function parses a text file into a pandas DataFrame. The text file should contain sequences of words and
    emojis. Each line in the text file should contain a word and an emoji separated by a space. Empty lines indicate
    the end of a sequence. The function also takes a vocabulary file for words and emojis. The vocabulary files
    should contain one word or emoji per line. The function returns a DataFrame where each row represents a sequence.
    The DataFrame has two columns: 'sequence_words' and 'sequence_emojis'. Each entry in 'sequence_words' is a numpy
    array of integers representing the words in the sequence. Each entry in 'sequence_emojis' is a numpy array of
    integers representing the emojis in the sequence. The integers are the indices of the words and emojis in the
    vocabulary files.

    Parameters: data_path (Path, optional): The path to the directory containing the data files. Defaults to the
    'data' directory in the parent directory of this file. file_path (Path, optional): The path to the text file to
    parse. Defaults to 'train.txt' in the data directory. vocab_file (str, optional): The name of the vocabulary file
    for words. Defaults to 'vocab.txt'. size_to_read (int, optional): The number of bytes to read from the text file.
    If 0, the entire file is read. Defaults to 0.

    Returns:
    DataFrame: A DataFrame where each row represents a sequence of words and emojis.
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / 'data'
    if file_path is None:
        file_path = data_path / 'train.txt'
    vocab_path = data_path / vocab_file
    emoji_path = data_path / 'emojis.txt'

    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    sequences = []

    with open(file_path, "r", encoding='utf-8') as f:
        if size_to_read:
            lines = f.readlines(size_to_read)
        else:
            lines = f.readlines()
        sequence_words = []
        sequence_emojis = []
        for line in lines:
            line = line.strip()
            if not line:
                sequences.append([np.asarray(sequence_words, dtype=np.int32),
                                  np.asarray(sequence_emojis, dtype=np.int32)])
                sequence_words = []
                sequence_emojis = []
            elif len(line.split()) == 2:
                if line.split()[0].lower() in word_vocab:
                    if line.split()[1] in emoji_vocab:
                        sequence_words.append(
                            word_vocab[line.split()[0].lower()])
                        sequence_emojis.append(emoji_vocab[line.split()[1]])

    df = pd.DataFrame(sequences, columns=['sequence_words', 'sequence_emojis'])

    return df
