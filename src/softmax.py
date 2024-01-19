import numpy as np
import metrics

def softmax(z: np.ndarray) -> np.ndarray:
    """
    This function is an implementation of the softmax function

    Parameters
    ----------
    z: np.ndarray
        Vector or value to compute softmax

    Return value
    ------------
        Return the result of the softmax function
    """
    assert len(z.shape) == 2, "x must be 2d"
    # Usefull to avoid overflow. Don't impact the result, because it's like if we have multiply the result by e^-max/e^-max, so by 1...
    norm = z - np.max(z, axis = 1).reshape(-1, 1)
    return np.exp(norm)/np.sum(np.exp(norm), axis = 1).reshape(-1, 1)


def gradient(X: np.ndarray, theta: np.ndarray, Y: np.ndarray):
    """
    This function is the gradient of the negative log likelihood of softmax

    Parameters
    ----------
    X: np.ndarray
        Matrix of known datas
    theta: np.ndarray
        Weights and bias
    Y: np.ndarray
        Array of known labels
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
    assert len(X.shape) == 2, "X must be 2d !"

    # Creation of X_hat with a column of ones added
    if X.shape[1] == theta.shape[1]:
        X_hat = X
    else:
        X_hat = np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def f(labels: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function compute a matrix witch is equal to 1 if y[i] = column of line i of the matrix
        """
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        result = np.zeros((len(y), len(labels)))
        for i in range(len(y)):
            result[i, y[i]] = 1
        return result

    labels = np.arange(theta.shape[0])
    gradient_result = np.zeros((theta.shape))

    tmp = f(labels, Y) - softmax(X_hat @ theta.T)
    for i in range(theta.shape[0]):
        gradient_result[i] = np.sum(tmp[:, i].reshape(-1, 1) * X_hat, axis = 0)
    gradient_result = -gradient_result
    return gradient_result

def gradient_descent(df, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int) -> np.ndarray:
    """This function implements the gradient descent. It iteratively computes the optimal parameters that minimize the given function.

    Parameters
    ----------
    `df`: function
        derivative function (i.e. gradient)
    `X`: np.ndarray
        Matrix of known parameters
    `Y`: np.ndarray
        Vector of known labels
    `theta`: np.ndarray
        Parameters to minimise
    `alpha`: float
        define how the function will converge. Values too big will give bad results and values too small won't converge or converge will too slowly
    `num_iters`: int
        Number of iterations

    Return value
    ------------
    Optimal parameters for the initial configuration and parameters"""
    for _ in range(num_iters):
        theta -= alpha * df(X, theta, Y)
    return theta

def train_log_reg_2(X: np.ndarray, Y: np.ndarray, theta: np.ndarray, num_iter: int, learning_rate: float) -> np.ndarray:
    """
    Function to train parameters of a logistic multinomial regression

    Parameters
    ----------
    X: np.ndarray
        Matrix of known parameters
    Y: np.ndarray
        Vector of known labels
    theta: np.ndarray
        Matrix of initial weight and bias to optimize
    num_iters: int
        Number of iterations
    learning_rate: float
        Speed of convergence of the gradient descent

    Return value
    ------------
    The optimal weights and bias for the datas and initial parameters
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    return gradient_descent(gradient, X, Y, theta, learning_rate, num_iter)

def predict_log_reg_2(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Function to predict values thanks to logistic multinomial regression

    Parameters
    ----------
    X: np.ndarray
        Matrix of known parameters
    theta: np.ndarray
        Matrix of weights and bias

    Return value
    ------------
    Return the predicted labels
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)
    if X.shape[1] == theta.shape[0]:
        X_hat = X
    else:
        X_hat = np.append(X, np.ones((X.shape[0], 1)), axis=1)

    results = softmax(X_hat @ theta.T)
    prediction = results == np.max(results, axis = 1).reshape(-1, 1)
    return np.where(prediction)[1]

def predict_compute_metrics(X_test, Y_test, theta):
    """
    Function to compute metrics for a set of data and weights and bias given

    Parameters
    ----------
    X_test: np.ndarray
        Set of datas to predict labels
    Y_test: np.ndarray
        Real prediction of the labels
    theta: np.ndarray
        Weights and bias

    Return value
    ------------
    Return the metrics computed
    """
    if not isinstance(X_test, np.ndarray):
        X_test = np.asarray(X_test)
    predicted_val_logreg = predict_log_reg_2(X_test, theta)
    return metrics.compute_metrics(Y_test, predicted_val_logreg)

def test_log_reg_with_dataset_values():
    """
    Function to test logistic multinomial regression with dataset values
    """
    from main import FEAT, LABELS

    X = np.asarray(FEAT)
    _, n = X.shape
    # I see that the performances are better for a theta initialised to 0 whereas a random theta...
    #theta = np.random.randint(1,10, size=(np.unique(LABELS).shape[0], X.shape[1] + 1)).astype(float)
    theta = np.zeros((np.unique(LABELS).shape[0], X.shape[1] + 1))
    n_it, lr = 1000, 1e-4

    theta = train_log_reg_2(X, LABELS, theta, n_it, lr)
    print("\n\nFound weights and bias:")
    print(theta)
    print()



def test_log_reg_f1score():
    """Test the efficiency of logistic regression with optimally chosen (before) parameters"""
    from main import FEAT, LABELS, FEAT_test, LABELS_test

    X = np.asarray(FEAT)
    _, n = X.shape
    theta = np.zeros((np.unique(LABELS).shape[0], X.shape[1] + 1))
    n_it, lr = 1000, 1e-4
    theta = train_log_reg_2(X, LABELS, theta, n_it, lr)
    print(predict_compute_metrics(FEAT_test, LABELS_test, theta))

if __name__ == "__main__":
    test_log_reg_with_dataset_values()
    test_log_reg_f1score()
