import numpy as np

def gradient_function(x: float):
    return np.cos(np.pi * (x + 1)) - x * np.pi * np.sin(np.pi * (x + 1))


def gradient_descent(gradient, MyVector, alpha: float, num_iters: int):
    """
    This function implements the gradient descent. It compute the optimal parameters MyVector.

    Parameters
    ----------
    gradient:
        derivate function
    MyVector:
        Initial vector of parameters to optimize
    alpha: float
        define how the function will converge. Too big values give bad results and too smal values not converge or converge too slowly
    num_iters: int
        Number of iterations
    Return value
    ------------
    Optimal vector for the initial configuration and parameters
    """
    for i in range(num_iters):
        MyVector -= alpha * gradient(MyVector)
    return MyVector
