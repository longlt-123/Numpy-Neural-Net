import numpy as np
from functions.utilities import convert_targets

def accuracy_score(y_true, y_pred):
    y_true = convert_targets(y_true, to="labels")
    y_pred = convert_targets(y_pred, to="labels")

    return np.sum(y_true == y_pred) / y_true.shape[0]

def precision_score(y_true, y_pred):
    y_true = convert_targets(y_true, to="labels")
    y_pred = convert_targets(y_pred, to="labels")

    true_positive = np.sum(y_true * y_pred, axis=-1)
    false_positive = np.sum((1 - y_true) * y_pred, axis=-1)

    precision = true_positive / (true_positive + false_positive + 1e-12)
    return np.mean(precision, axis=0)

def recall_score(y_true, y_pred):
    y_true = convert_targets(y_true, to="labels")
    y_pred = convert_targets(y_pred, to="labels")

    true_positive = np.sum(y_true * y_pred, axis=-1)
    false_negative = np.sum((1 - y_pred) * y_true, axis=-1)

    recall = true_positive / (true_positive + false_negative + 1e-12)
    return np.mean(recall, axis=0)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return f1