from sklearn.metrics import f1_score, confusion_matrix
import numpy as np


def evaluate_predictions(preds, y_test):
    """
    Evaluates the predictions made by a model.

    Parameters
    ----------
    preds : array-like
        The predictions made by the model.
    y_test : array-like
        The actual labels.

    Returns
    -------
    dict
        A dictionary containing the weighted accuracy, weighted F1 score, and confusion matrix.
    """
    return dict(weighted_accuracy=np.mean(preds == y_test),
                weighted_f1_score=f1_score(y_test, preds, average='weighted'),
                confusion_matrix=confusion_matrix(y_test, preds))
