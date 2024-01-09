import numpy as np
import pytest
from pprint import pprint

from gradient_descent import grad_desc_ml

# NB: floating is any (numpy) floating type NDArray or not
from numpy import floating as fl, int32 as i32
from numpy.random import rand, randint
from numpy.typing import NDArray
from pandas import DataFrame
import pandas as pd
import metrics

# NOTE: parameters found after testing with different values, see branch "gpu-training" of repo for more details
best_w, best_b = np.array([0.53452349, 0.36463584, 1.16132476, 1.08204578]), 0.45146791

def z(X: NDArray, w: NDArray, b: float) -> fl:
    """
    Returns
    -------
    ``np.dot(X, w) + b``: `float` or `NDArray[float]` (i.e. `floating`)
    Notes
    -----
    `w` and `X` can be interchanged e.g. `z(X, w, b)`, it won't give
    the same result (in general) but as long as matrix multiplication dimensions
    are respected, it will work."""
    return np.dot(X, w) + b


def sigmoid(z: fl) -> fl:
    """Returns
    -----------
    1 / (1 + exp(-z))"""
    return 1 / (1 + np.exp(-z))


def norm(X: NDArray) -> NDArray:
    return (X - np.mean(X)) / np.std(X)


def grad(X: NDArray, y: NDArray, w: NDArray, b: float) -> tuple:
    """Computes (vectorized) the gradient of the log loss function w.r.t "w" and "b" for the current iteration.
    It is used in the gradient descent algorithm.

    Parameters
    ----------
    `X` : NDArray
        Samples / features.
    `y` : NDArray
        labels / class associated to each sample.
    `w` : NDArray
        weights vector.
    `b` : float
        bias
    Returns
    -------
    (dw, db) :
        The gradient of the log loss function w.r.t "w" and "b"."""

    predictions = sigmoid(z(X, w, b))  # Sigmoid function applied to z
    errors = y - predictions  # Difference between actual and predicted values
    db = -np.sum(errors)  # Vectorized computation of db component

    X_sum_over_rows = np.sum(X, axis=1)  # Sum over rows of X
    dw = -np.sum(X_sum_over_rows * errors)  # Vectorized computation of dw component
    return dw, db


def train_log_reg(X: NDArray | DataFrame, y: NDArray, w: NDArray, b: float, n_it: int, lr: float) -> tuple[NDArray, float]:
    """
    Parameters
    ----------
    `X` : NDArray
        Samples / features.
    `y` : NDArray
        labels / class associated to each sample.
    `w` : NDArray
        initial weight vector.
    `ab` : float
        inital bias
    `n_it` : int
        iterations number
    `lr` : float
        learning rate
    Returns
    -------
        Trained (weight vector, bias) with gradient descent that minimize the log loss function."""
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    return grad_desc_ml(X, y, grad, w, b, lr, n_it)


def predict_log_reg(X: NDArray, w: NDArray, b):
    """Predict the class labels for a set of examples X using logistic regression parameters w and b.
    :param X: The input features. 2D Matrix NDArray
    :param w: The weights of the logistic regression model. Vector NDArray
    :param b: The bias of the logistic regression model. float
    :return: Vector of predicted class labels (0 or 1) for each example in X. Vector NDArray
    """
    from main import LAB_VAL_IDX

    # return i32(sigmoid(z(X, w, b)).get() >= 0.5)
    # print(sigmoid(z(X, w, b)))
    # predicted= sigmoid((z(X, w, b))).get()
    # if DataFrame or Series is passed, convert to numpy array
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    predicted = sigmoid(norm(z(X, w, b)))  # type: ignore
    class_nb = len(LAB_VAL_IDX)
    return i32(predicted * class_nb)  # returns k-1 if val < (k/class_nb) i.e. 0 if < 1/3, 1 if < 2/3...


def predict_compute_metrics(X_test, Y_test, w, b):
    if not isinstance(X_test, np.ndarray):
        X_test = np.asarray(X_test)
    predicted_val_logreg = predict_log_reg(X_test, w, b)
    # metrics = compute_metrics(Y_test, predicted_val_logreg)
    # return metrics


# ================================================================
# ======================= TEST:==================================
# ================================================================


@pytest.mark.parametrize("m, n", randint(0, 70, size=(20, 2)))
def test_log_reg_runs_with_random_values(m, n):
    X, y, w, b = rand(m, n), rand(m), rand(n), rand()
    n_it, lr = 100, 0.03
    w, b = train_log_reg(X, y, w, b, n_it, lr)


def test_log_reg_runs_with_dataset_values():
    from main import FEAT, LABELS

    X = np.asarray(FEAT)
    m, n = X.shape
    init_w = np.random.rand(n)
    init_b = np.random.rand()
    n_it, lr = 1000, 1e-5

    w, b = train_log_reg(X, LABELS, init_w, init_b, n_it, lr)
    print("\n\nFound weights and bias:")
    pprint(w)
    pprint(b)
    print()


def test_log_reg_f1score():
    """Test the efficiency of logistic regression with optimally chosen (before) parameters"""
    from main import FEAT_test, LABELS_test

    # NOTE: parameters found after testing with different values, see branch "gpu-training" for more details
    w, b = np.array([0.53452349, 0.36463584, 1.16132476, 1.08204578]), 0.45146791
    # pred_compute(FEAT_test, LABELS_test, w, b)

    predicted_val_logreg = predict_log_reg(np.asarray(FEAT_test), w, b)
    # TODO: replace with our impl of f1_score
    from sklearn.metrics import f1_score

    score = metrics.f1_score(pd.DataFrame(LABELS_test), predicted_val_logreg)
    score2 = f1_score((LABELS_test), predicted_val_logreg, average="micro")
    print("weights:", w, "\nbias:", b, "\n")
    print("F1 score:", score, "\n")
    print("F1 score:", score2, "\n")
