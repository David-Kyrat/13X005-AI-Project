from pprint import pprint
from typing import Any

import numpy as np
from main import COL_NAMES, DATASET_ID, FEAT, LABELS  # noqa: F401
from gradient_descent import grad_desc_ml
from numpy import float32 as f32

# NB: floating is any (numpy) floating type NDArray or not
from numpy import floating as fl
from numpy.typing import NDArray
from pandas import DataFrame


def z(X: NDArray, w: NDArray, b: fl) -> fl:
    """:return: ``np.dot(X, w) + b``.
    float or NDArray[float] (i.e. `fl`)
    NOTE: `w` and `X` can be interchanged e.g. (`z(X, w, b)), it won't give
    the same result (in general) but as long as matrix multiplication dimensions
    are respected, it will work."""
    return np.dot(X, w) + b


def sigmoid(z: fl):
    return 1 / (1 + np.exp(-z))


def norm(X: NDArray):
    return (X - np.mean(X)) / np.std(X)


def grad(X: NDArray, y: NDArray, w: NDArray, b: fl):
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
    `b` : fl (float or NDArray[float])
        bias
    Returns
    -------
    (dw, db) :
        The gradient of the log loss function w.r.t "w" and "b"."""

    predictions = sigmoid(z(w, X, b))  # Sigmoid function applied to z
    errors = y - predictions  # Difference between actual and predicted values
    db = -np.sum(errors)  # Vectorized computation of db component

    X_sum_over_rows = np.sum(X, axis=1)  # Sum over rows of X
    dw = -np.sum(X_sum_over_rows * errors)  # Vectorized computation of dw component

    return dw, db


def train_log_reg(X: NDArray, y: NDArray, w: NDArray, b: fl, n_it: int, lr: float) -> tuple[NDArray, fl]:
    """
    Parameters
    ----------
    `X` : NDArray
        Samples / features.
    `y` : NDArray
        labels / class associated to each sample.
    `w` : NDArray
        initial weight vector.
    `b` : fl (float or NDArray[float])
        inital bias
    `n_it` : int
        iterations number
    `lr` : float
        learning rate
    Returns
    -------
    Trained (weight vector, bias) with gradient descent that minimize the log loss function."""
    return grad_desc_ml(X, y, grad, w, b, lr, n_it)
