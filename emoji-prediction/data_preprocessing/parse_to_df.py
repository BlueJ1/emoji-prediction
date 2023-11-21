from pathlib import Path

import numpy as np
import pandas as pd


def parse_to_df(data_path: Path = None, file_path: Path = None, size_to_read: int = 0):
    if data_path is None:
        data_path = Path(__file__).parent.parent / 'data'
    if file_path is None:
        file_path = data_path / 'train.txt'

    vocab_path = data_path / 'vocab.txt'
    emoji_path = data_path / 'emojis.txt'

    with open(vocab_path, 'r') as f:
        word_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    with open(emoji_path, 'r') as f:
        emoji_vocab = {w[:-1]: i for i, w in enumerate(f.readlines())}

    sequences = []

    with open(file_path, "r") as f:
        if size_to_read:
            lines = f.readlines(size_to_read)
        else:
            lines = f.readlines()
        sequence_i = 0
        sequence_words = []
        sequence_emojis = []
        for line in lines:
            line = line.strip()
            if not line:
                sequences.append([np.asarray(sequence_words, dtype=np.int32),
                                  np.asarray(sequence_emojis, dtype=np.int32)])
                sequence_i += 1
                sequence_words = []
                sequence_emojis = []
            elif len(line.split()) == 2 and line.split()[0].lower() in word_vocab and line.split()[1] in emoji_vocab:
                sequence_words.append(word_vocab[line.split()[0].lower()])
                sequence_emojis.append(emoji_vocab[line.split()[1]])

    df = pd.DataFrame(sequences, columns=['sequence_words', 'sequence_emojis'])

    return df
