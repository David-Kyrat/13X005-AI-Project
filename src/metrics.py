# TODO : COMMENT
# TODO : Adapt to the logistic regression implementation (columns names...)

def evaluation(true_labels, predict_labels, label):
    """
    This function evaluates a prediction by calculating the true/false positive/negative, which are necessary for the below metrics.

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels

    Return value
    ------------
    tuple of int (4 elements wich are the true/false positive/negative values)
    """
    true_negative = sum(real == label and pred != label for real, pred in zip(true_labels, predict_labels))
    true_positive = sum(real == label and pred == label for real, pred in zip(true_labels, predict_labels))
    false_negative = sum(real == label and pred != label for real, pred in zip(true_labels, predict_labels))
    false_positive = sum(true != label and pred == label for true, pred in zip(true_labels, predict_labels))

    return true_positive, false_positive, true_negative, false_negative


def accuracy(true_labels, predicted_labels):
    """
    This function implements the accuracy metric wich measure the proportion of correct predictions considering all predictions

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels

    Return value
    ------------
    float number wich is the accuracy proportion value between 0 and 1.
    """

    correct_predictions = sum(true == pred for true, pred in zip(true_labels, predicted_labels))
    total_instances = len(true_labels)
    return correct_predictions / total_instances


def precision(true_labels, predict_labels):
    """
    This function implements the precision metric wich measure the proportion of false positive predictions considering every positive predictions for each label, and returns the average precision

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels

    Return value
    ------------
    float number wich is the precision value between 0 and 1.
    """

    labels_names = list(set(true_labels))
    precisions = []
    for label in labels_names:
        true_positive, false_positive, _, _ = evaluation(true_labels, predict_labels, label)
        tp_fp = true_positive + false_positive
        precisions.append(0 if tp_fp == 0 else true_positive / tp_fp)

    return sum(precisions) / len(labels_names)


def recall(true_labels, predict_labels):
    """
    This function implements the precision metric wich measure the proportion of uncorrect negatives predictions (recall of 1 means the number of fale negatives is null) for each label, and returns the average recall

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels

    Return value
    ------------
    float number wich is the recall value between 0 and 1.
    """

    labels_names = list(set(true_labels))
    recalls = []
    for label in labels_names:
        true_positive, _, _, false_negative = evaluation(true_labels, predict_labels, label)
        tp_fn = true_positive + false_negative
        recalls.append(0 if tp_fn == 0 else true_positive / tp_fn)

    return sum(recalls) / len(labels_names)


def f1_score(true_labels, predict_labels):
    """
    This function implements the f1 score metric wich is a balanced measure between precision and recall that measures errors more broadly

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels

    Return value
    ------------
    float number wich is the f1 score value between 0 and 1.
    """

    prec = precision(true_labels, predict_labels)
    rec = recall(true_labels, predict_labels)
    prec_rec = prec + rec
    return 0 if prec_rec == 0 else 2 * (prec * rec) / (prec + rec)


def compute_metrics(true_labels, predicted_labels):
    """This function calculates the performance metrics for each class in a binary classification problem.
    The metrics calculated are Precision, Recall, and F1 Score.
    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels
    Returns
    -------
    dict: A dictionary containing the performance metrics for each class."""

    _precision = precision(true_labels, predicted_labels)
    _recall = recall(true_labels, predicted_labels)
    _accuracy = accuracy(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return {"precision": _precision, "recall": _recall, "accuracy": _accuracy, "f1_score": f1}


# ================================================================
# ======================= TEST:==================================
# ================================================================


import sklearn.metrics as sk_metrics
from numpy.testing import assert_almost_equal
from pandas import DataFrame  # noqa: F401
import numpy as np  # noqa: F401


def test_metrics():
    from main import FEAT_test, LABELS_test
    import naive_bayes as nb  # noqa: F401
    import log_reg as lr
    from pprint import pprint  # noqa: F401

    # predictions = nb.predict_bayes_all(FEAT_test)
    predictions = lr.predict_log_reg(FEAT_test.to_numpy(), lr.best_w, lr.best_b)
    # print("predictions : ", predictions
    # print("\n")

    # TEST :

    true_labels = LABELS_test
    # print("Reality :", true_labels)
    # print("\nevals :", [evaluation(true_labels, predictions, label) for label in labels_train.unique()])

    obtained = accuracy(true_labels, predictions)
    expected = sk_metrics.accuracy_score(true_labels, predictions)
    decimal = 2

    print("\nAccuracy : ", obtained)
    print("sklearn accuracy : ", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)

    obtained = precision(true_labels, predictions)
    expected = sk_metrics.precision_score(true_labels, predictions, average="macro")
    print("\nPrecision : ", obtained)
    print("sklearn precision : ", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)

    obtained = recall(true_labels, predictions)
    expected = sk_metrics.recall_score(true_labels, predictions, average="macro")
    print("\nRecall : ", obtained)
    print("sklearn recall : ", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)

    obtained = f1_score(true_labels, predictions)
    expected = sk_metrics.f1_score(true_labels, predictions, average="macro")
    print("\nF1 Score : ", obtained)
    print("sklearn F1 Score :", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)


def test_compute_metrics():
    from main import FEAT_test, LABELS_test
    import log_reg as lr
    from pprint import pprint
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    import pytest

    predictions = lr.predict_log_reg(FEAT_test.to_numpy(), lr.best_w, lr.best_b)
    computed_metrics = compute_metrics(LABELS_test, predictions)

    precision = precision_score(LABELS_test, predictions, average="macro")
    recall = recall_score(LABELS_test, predictions, average="macro")
    accuracy = accuracy_score(LABELS_test, predictions)
    f1 = f1_score(LABELS_test, predictions, average="macro")

    computed_metrics_expected = {
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
    }
    pprint(computed_metrics)

    assert computed_metrics == pytest.approx(computed_metrics_expected, 0.07)
