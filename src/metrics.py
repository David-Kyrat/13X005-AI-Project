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
    from main import FEAT, LAB_NAME, DATA_train, LABELS_STR_train
    from naive_bayes import get_distrib_parameters, predict_bayes

    # Extracting data :
    # labels_train: DataFrame = DATA_train[LAB_NAME]  # type: ignore
    # labels_test: DataFrame = DATA_test[LAB_NAME]  # type: ignore

    # "TRAIN" :
    parameters = get_distrib_parameters(FEAT, LABELS_STR_train)

    # predicitons = [predict_bayes(FEAT.iloc[idx], parameters) for idx in range(len(FEAT))]
    predicitons = [predict_bayes(sample[1:], parameters) for sample in FEAT.itertuples()]
    # print("predictions : ", predicitons
    # print("\n")

    # TEST :

    true_labels = list(LABELS_STR_train)
    # print("Reality :", true_labels)
    # print("\nevals :", [evaluation(true_labels, predicitons, label) for label in labels_train.unique()])

    obtained = accuracy(true_labels, predicitons)
    expected = sk_metrics.accuracy_score(true_labels, predicitons)
    decimal = 2

    print("\nAccuracy : ", obtained)
    print("sklearn accuracy : ", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)

    obtained = precision(true_labels, predicitons)
    expected = sk_metrics.precision_score(true_labels, predicitons, average="macro")
    print("\nPrecision : ", obtained)
    print("sklearn precision : ", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)

    obtained = recall(true_labels, predicitons)
    expected = sk_metrics.recall_score(true_labels, predicitons, average="macro")
    print("\nRecall : ", obtained)
    print("sklearn recall : ", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)

    obtained = f1_score(true_labels, predicitons)
    expected = sk_metrics.f1_score(true_labels, predicitons, average="macro")
    print("\nF1 Score : ", obtained)
    print("sklearn F1 Score :", expected)
    assert_almost_equal(obtained, expected, decimal=decimal)
