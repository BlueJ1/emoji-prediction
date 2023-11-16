from pathlib import Path
import numpy as np

data_path = Path(__file__).parent.parent / 'data'
train_path = data_path / 'train.txt'

vocab = {'': 0, 'UNK': 1, '<START>': 2, '<END>': 3}
def load_glove_embeddings(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        embeddings_index = {}
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        return embeddings_index

# Replace 'glove.6B.100d.txt' with the path to the specific GloVe file you downloaded
glove_embeddings = load_glove_embeddings('glove.6B.50d.txt')

with open(train_path, 'r') as f:
    lines = f.readlines()

    for line in lines:
        split = line.split()
        if len(split) == 2:

