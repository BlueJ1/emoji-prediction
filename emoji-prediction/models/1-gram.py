import numpy as np
from pathlib import Path


def one_gram_predict(words):
    data_path = Path(__file__).parent.parent / 'data'
    matrix_file = data_path / 'one_gram_matrix.npy'

    one_gram_matrix = np.load(matrix_file)

    predictions = []
    for word in words:
        predictions.append(np.argmax(one_gram_matrix[word]))
    return predictions
