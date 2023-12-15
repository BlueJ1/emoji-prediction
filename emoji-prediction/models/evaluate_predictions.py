from sklearn.metrics import f1_score
import numpy as np


def evaluate_predictions(preds, y_test):
    return dict(average_accuracy=np.mean(preds == y_test),
                weighted_f1_score=f1_score(y_test, preds, average='weighted'))
