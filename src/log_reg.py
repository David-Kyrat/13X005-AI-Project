import numpy as np
import pytest

# from main import COL_NAMES, DATASET_ID, FEAT, LABELS  # noqa: F401
from gradient_descent import grad_desc_ml

# NB: floating is any (numpy) floating type NDArray or not
from numpy import floating as fl
from numpy.random import rand, randint
from numpy.typing import NDArray


def z(X: NDArray, w: NDArray, b: float) -> fl:
    """
    Returns
    -------
    ``np.dot(X, w) + b``: `float` or `NDArray[float]` (i.e. `floating`)
    Notes
    -----
    `w` and `X` can be interchanged e.g. `z(w, X, b)`, it won't give
    the same result (in general) but as long as matrix multiplication dimensions
    are respected, it will work."""
    return np.dot(X, w) + b


def sigmoid(z: fl) -> fl:
    """Returns
    -----------
    1 / (1 + exp(-z))"""
    return 1 / (1 + np.exp(-z))


def norm(X: NDArray):
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


def train_log_reg(X: NDArray, y: NDArray, w: NDArray, b: float, n_it: int, lr: float) -> tuple[NDArray, float]:
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
    for _ in range(n_it):
        grad_w, grad_b = grad(X, y, w, b)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

    # return grad_desc_ml(X, y, grad, w, b, lr, n_it)


@pytest.mark.parametrize("m, n", randint(0, 70, size=(20, 2)))
def test_log_reg_with_random_values(m, n):
    X, y, w, b = rand(m, n), rand(m), rand(n), rand()
    n_it, lr = 100, 0.03
    w, b = train_log_reg(X, y, w, b, n_it, lr)

def test_log_reg_with_dataset_values():
    from main import FEAT, LABELS
    m, n = FEAT.shape
    init_w = np.random.rand(n)
    init_b = np.random.rand()
    n_it, lr = 1000, 1e-5

    w, b = train_log_reg(FEAT.to_numpy(), LABELS, init_w, init_b, n_it, lr)
    print(w, b)
    print("Found weights and bias:\n\tw =", w, "\tb =", b)
