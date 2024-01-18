import numpy as np
import metrics

# def softmax(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
# 
#     if not isinstance(X, np.ndarray):
#         X = np.asarray(X)
#     if not isinstance(theta, np.ndarray):
#         theta = np.asarray(theta)
#     if len(X.shape) != 2:
#         print("X must be 2d array")
#         print("I suppose that X is 1d dimension, so I convert it into 2d dimension")
#         X = X.reshape(1, -1)
#     if X.shape[1] == theta.shape[0]:
#         X_hat = X
#     else:
#         X_hat = np.append(X, np.ones((X.shape[0], 1)), axis=1)
# 
#     K = np.sum(np.exp(X_hat @ theta), axis=1)
# 
#     K = 1/K
# 
#     return K[:, None]*np.exp(X_hat @ theta) 

def softmax(x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    norm = x - np.max(x, axis = 1).reshape(-1, 1)
    return np.exp(norm)/np.sum(np.exp(norm), axis = 1).reshape(-1, 1)


theta = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [3, 2, 4, 7]])

X = np.array([[0.5, 0.7, 0.8], [0.4, 0.6, 0.3], [0.5, 0.9, 0.2]])
Y = [0, 2, 1]

#print(softmax(X, theta.T))
#print(softmax([0.5, 0.7, 0.8], theta.T))

def gradient(X: np.ndarray, theta: np.ndarray, Y: np.ndarray):
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
    if len(X.shape) != 2:
        print("X must be 2d array")
        print("I suppose that X is 1d dimension, so I convert it into 2d dimension")
        X = X.reshape(1, -1)
    if X.shape[1] == theta.shape[0]:
        X_hat = X
    else:
        X_hat = np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def f(labels: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        result = np.zeros((len(y), len(labels)))
        for i in range(len(y)):
            result[i, y[i]] = 1
        return result

    labels = np.arange(theta.shape[1])
    gradient_result = np.zeros((theta.shape))
    gradient_result = gradient_result.T
        # V1
        #for i in range(theta.shape[0]):
        #    gradient_result[i] += X_hat[n] * (f(Y[n], i) - softmax(X_hat[n], theta, i)) 
        # V2
    tmp = f(labels, Y) - softmax(X_hat @ theta)
    for i in range(theta.shape[1]):
        gradient_result[i] = np.sum(tmp[:, i].reshape(-1, 1) * X_hat, axis = 0)

    return gradient_result

# print(gradient(X, theta.T, Y))

def gradient_descent(df, alpha: float, num_iters: int, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """This function implements the gradient descent. It iteratively computes the optimal parameters that minimize the given function.

    Parameters
    ----------
    `df`: function
        derivative function (i.e. gradient)
    `alpha`: float
        define how the function will converge. Values too big will give bad results and values too small won't converge or converge will too slowly
    `num_iters`: int
        Number of iterations
    `*arg`:
        All the argument passed to the gradient function

    Return value
    ------------
    Optimal vector for the initial configuration and parameters"""
    for _ in range(num_iters):
        theta -= alpha * df(X, theta.T, Y)
    return theta

def train_log_reg_2(X: np.ndarray, Y: np.ndarray, theta: np.ndarray, num_iter: int, learning_rate: float) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)


    return gradient_descent(gradient, learning_rate, num_iter, X, Y, theta)

# print(train_log_reg_2(X, Y))

def predict_log_reg_2(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)
    if X.shape[1] == theta.shape[0]:
        X_hat = X
    else:
        X_hat = np.append(X, np.ones((X.shape[0], 1)), axis=1)

    results = softmax(X_hat @ theta.T)
    print(results)
    prediction = results == np.max(results, axis = 1).reshape(-1, 1)
    print(prediction)
    return np.where(prediction)[1]

# print(predict_log_reg_2(X, train_log_reg_2(X, Y)))

def predict_compute_metrics(X_test, Y_test, w, b):
    if not isinstance(X_test, np.ndarray):
        X_test = np.asarray(X_test)
    predicted_val_logreg = predict_log_reg(X_test, w, b)
    return metrics.compute_metrics(Y_test, predicted_val_logreg)

def test_log_reg_with_dataset_values():
    from main import FEAT, LABELS

    X = np.asarray(FEAT)
    _, n = X.shape
    theta = np.random.randint(1,10, size=(np.unique(LABELS).shape[0], X.shape[1] + 1)).astype(float)
    theta = np.random.random( size=(np.unique(LABELS).shape[0], X.shape[1] + 1)).astype(float)
    n_it, lr = 1000, 1e-5

    theta = train_log_reg_2(X, LABELS, theta, n_it, lr)
    print("\n\nFound weights and bias:")
    print(theta)
    print()
    return theta


from main import FEAT, LABELS, FEAT_test, LABELS_test
theta = test_log_reg_with_dataset_values()
print(predict_log_reg_2(FEAT_test, theta))
exit()
X = np.asarray(FEAT)
learning_rate = np.linspace(0.1, 1e-5, 10)
index = 0
maximum = 0
for i in range(len(learning_rate)):
    theta = np.random.randint(1,10, size=(np.unique(LABELS).shape[0], X.shape[1] + 1)).astype(float)
    theta = train_log_reg_2(X, LABELS, theta, 1000, learning_rate[i])
    predicted_val_logreg = predict_log_reg_2(np.asarray(FEAT_test), theta)
    scores = metrics.compute_metrics(LABELS_test, predicted_val_logreg)
    if scores['f1_score'] > maximum:
        maximum = scores['f1_score']
        index = i
print(maximum)
print(index)

def test_log_reg_f1score():
    """Test the efficiency of logistic regression with optimally chosen (before) parameters"""
    from main import FEAT, LABELS, FEAT_test, LABELS_test

    X = np.asarray(FEAT)
    _, n = X.shape
    index = 0
    maximum = 0
    n_it = 1000
    lr = 1e-5
    for i in range(5):
        theta = train_log_reg_2(X, LABELS, theta, n_it, lr)
        predicted_val_logreg = predict_log_reg_2(np.asarray(FEAT_test), theta)
        scores = metrics.compute_metrics(LABELS_test, predicted_val_logreg)
        if scores['f1_score'] > maximum:
            index = i
        lr *= 10
    print(maximum)


