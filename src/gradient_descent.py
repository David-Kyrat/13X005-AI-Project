def gradient_descent(X: np.ndarray, y: np.ndarray, MyVector: np.ndarray, num_iters: int, alpha: float):
    """
    This function implements the gradient descent. It compute the optimal parameters MyVector.

    Parameters
    ----------
    X: np.ndarray
        Matrix of covariables
    y: np.ndarray
        Vector of labels
    MyVector: np.ndarray
        Initial vector of parameters to optimize
    num_iters: int
        Number of iterations
    alpha: float
        define how the function will converge. Too big values give bad results and too smal values not converge or converge too slowly

    Return value
    ------------
    Optimal vector for the initial configuration and parameters
    """
    for i in range(num_iters):
        MyVector -= alpha * gradient(X, y, MyVector)
    return MyVector
