import matplotlib.pyplot as plt
import numpy as np
import plot_util
from numpy import cos, pi, sin
from numpy import floating as fl
from numpy.typing import NDArray


def gradient_descent(df, params: NDArray, alpha: float, num_iters: int) -> NDArray:
    """This function implements the gradient descent. It iteratively computes the optimal parameters that minimize the given function.

    Parameters
    ----------
    `df`: function
        derivative function (i.e. gradient)
    `params`: NDArray
        Initial vector of parameters to optimize
    `alpha`: float
        define how the function will converge. Values too big will give bad results and values too small won't converge or converge will too slowly
    `num_iters`: int
        Number of iterations
    Return value
    ------------
    Optimal vector for the initial configuration and parameters"""
    for _ in range(num_iters):
        params -= alpha * df(params)
    return params


def grad_desc_ml(
    features: NDArray, labels: NDArray, df, w: NDArray, b: fl, alpha: float, num_iters: int
) -> tuple[NDArray, fl]:
    """Same gradient descent `gradient_desent` method, but that takes `features` (X) and `labels` (y)
    as additional parameters, since they're obviously going to be need for any kind of learning whatsoever.
    Parameters
    ----------
    `features` : NDArray
        Samples / features.
    `labels` : NDArray
        labels / class associated to each sample.
    `df`: function
        derivative function (i.e. gradient)
    `w` : NDArray
        weights vector.
    `b` : fl (float or NDArray[float])
        bias
    `alpha`: float
        define how the function will converge. Values too big will give bad results and values too small won't converge or will converge too slowly
    `num_iters`: Number of iterations
    Return value
    ------------
    Optimal vector for the initial configuration and parameters"""

    for _ in range(num_iters):
        grad_w, grad_b = df(features, labels, w, b)
        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b


def test_gradient_descent():
    """This function test the gradient descent implemented for `f(x) = x * cos(pi * (x + 1))` with 10000 iterations.
    and tests that it is close enough of `scipy.optimize.fmin`"""

    def f(x):
        return x * cos(pi * (x + 1))

    def df(x):
        return cos(pi * (x + 1)) - x * pi * sin(pi * (x + 1))

    x = np.linspace(-2 * pi, 2 * pi, 100)
    params = np.array([-pi, 0.0, pi])
    alpha, num_iters = 0.01, 10000
    optimal_x = gradient_descent(df, params, alpha, num_iters)

    from scipy.optimize import fmin  # to test the result

    expected_optimal_x = np.array([fmin(f, param, disp=False) for param in params])
    plot_util.plot_vs(
        x,
        f(x),
        plot_x2=optimal_x,
        plot_f2=f(optimal_x),
        title=f"local minima search with Gradient descent. {num_iters} iterations.",
        f1label=r"$f(x) = x * \cos(\pi  (x + 1))$",
        f2label="local minima",
        f1style="-k",
        f2style="ro",
        show=False,
    )
    plt.plot(expected_optimal_x, f(expected_optimal_x), "co", label="optimal x found by scipy.optimize.fmin", alpha=0.4)
    plt.legend()
    plt.savefig("res/3.1_gradient_descent_minima.png")
    plt.show()
    # to ensure test does not fail due to dimension mismatch
    expected_optimal_x = expected_optimal_x.reshape(optimal_x.shape)
    np.testing.assert_allclose(optimal_x, expected_optimal_x, atol=1e-4)
