import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import pi, sin, cos


def gradient_descent(df, params: NDArray, alpha: float, num_iters: int) -> NDArray:
    """
    This function implements the gradient descent. It iteratively computes the optimal parameters that minimize the given function.

    Parameters
    ----------
    `df`: function
        derivative function (i.e. gradient)
    `params`: NDArray
        Initial vector of parameters to optimize
    `alpha`: float
        define how the function will converge. Too big values give bad results and too small values won't converge or converge too slowly
    `num_iters`: int
        Number of iterations
    Return value
    ------------
    Optimal vector for the initial configuration and parameters
    """
    for _ in range(num_iters):
        params -= alpha * df(params)
    return params


# def test_gradient_descent(f, df, MyVector: NDArray, alpha: float, num_iters: int):
def test_gradient_descent():
    """This function test the gradient descent implemented for `f(x) = x * cos(pi * (x + 1))` with 1000 iterations."""

    def f(x):
        return x * cos(pi * (x + 1))

    def df(x):
        return cos(pi * (x + 1)) - x * pi * sin(pi * (x + 1))

    x = np.linspace(0, 2 * pi, 100)
    plt.figure()
    plt.plot(x, f(x))
    params = np.random.rand(1)
    alpha, num_iters = 1, 1000
    optimal_x = gradient_descent(df, params, alpha, num_iters)
    plt.plot(optimal_x, f(optimal_x), "ro")
    plt.show()

    # for test
    from scipy.optimize import fmin

