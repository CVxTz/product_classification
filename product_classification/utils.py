""" utility functions """
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_predicted):

    return {
        "f1_macro": f1_score(y_true, y_predicted, average="macro"),
        "f1_weighted": f1_score(y_true, y_predicted, average="weighted"),
        "accuracy": accuracy_score(y_true, y_predicted),
        "per_class_f1": f1_score(y_true, y_predicted, average=None).tolist(),
    }
