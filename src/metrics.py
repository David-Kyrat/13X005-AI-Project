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


# ================================================================
# ======================= TEST:==================================
# ================================================================


import sklearn.metrics as sk_metrics
from numpy.testing import assert_almost_equal
from pandas import DataFrame
import numpy as np

def test_metrics():
    from main import FEAT_test, LABELS_STR_test
    from naive_bayes import predict_bayes_all

    predictions = predict_bayes_all(FEAT_test)
    # print("predictions : ", predictions
    # print("\n")

    # TEST :

    true_labels = list(LABELS_STR_test)
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
