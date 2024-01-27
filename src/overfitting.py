import numpy as np
import pandas
import matplotlib.pyplot as plt
from softmax import train_log_reg_2, predict_log_reg_2, predict_compute_metrics
from naive_bayes import get_distrib_parameters, predict_bayes
from main import FEAT, FEAT_test, LABELS, LABELS_test
from metrics import f1_score


def get_percentage_of_data(feat: np.ndarray, labels: np.ndarray, percentage: int) -> np.ndarray:
    """
    Allow to get random sampling of a percentage of datas

    Parameters
    ----------
    feat: np.ndarray
        feat we want a percentage of
    labels: np.ndarray
        labels we want a percentage of
    percentage: int
        percentage of datas we want

    Return value
    ------------
    Return a tuple with the np.array with the right percentage of the datas we want,
    and with the np.array of the datas which weren't taken
    """

    if not isinstance(feat, np.ndarray):
        feat = np.asarray(feat)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    number_of_elements = np.round(len(feat) * percentage / 100).astype(int)
    elements_remain = len(feat) - number_of_elements
    indices = np.arange(number_of_elements).astype(int)
    indices_remain = np.arange(elements_remain).astype(int) + number_of_elements
    data = np.append(feat, labels[:, None], axis=1)
    np.random.shuffle(data)
    newfeat = data[indices, :-1]
    newlabels = data[indices, -1]
    feat_remain = data[indices_remain, :-1]
    labels_remain = data[indices_remain, -1]
    # Is it usefull to keep feat_remain and labels_remain ?
    return newfeat, newlabels.astype(int)

def add_noise_to_data(labels: np.ndarray, percentage: int) -> np.ndarray:
    """
    Allow to get random sampling of a percentage of datas

    Parameters
    ----------
    labels: np.ndarray
        labels we want a percentage of
    percentage: int
        percentage of datas we want

    Return value
    ------------
    Return a tuple with the np.array with the right percentage of the datas we want,
    and with the np.array of the datas which weren't taken
    """

    values = np.array(labels)
    index = np.random.binomial(1, percentage/100, len(values))
    newvalues = np.random.binomial(np.unique(values).shape[0] - 1, 0.5, len(values))
    newvalues *= index
    invert_index = np.zeros(index.shape) == index

    values *= invert_index

    values += newvalues

    return values

def overfitting_naive_bayes(feat: np.ndarray, labels: np.ndarray, feat_test: np.ndarray, labels_test: np.ndarray):
    """
    Function to show the overfitting.

    This function will test the performances of the model for a too small set of datas
    """
    if not isinstance(feat, np.ndarray):
        feat = np.asarray(feat)
    if not isinstance(feat_test, np.ndarray):
        feat_test = np.asarray(feat_test)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    if not isinstance(labels_test, np.ndarray):
        labels_test = np.asarray(labels_test)
   
    percentage = np.arange(5, 100)

    percent_10 = [10 for i in range(5,100)]
    percent_100 = [100 for i in range(5,100)]
    index = np.arange(5,100)

    f1_score_test = []
    f1_score_training = []
    f1_score_test_10 = []
    f1_score_training_10 = []
    f1_score_test_100 = []
    f1_score_training_100 = []
    f1_score_noise_test = []
    f1_score_noise_training = []

    for i in range(len(percentage)):

        # F1 score for 5% -> 100% of datas
        training_feat, training_labels = get_percentage_of_data(feat, labels, percentage[i])
        params = get_distrib_parameters(training_feat, training_labels)
        predicted_val_naive_bayes_test = []
        for j in feat_test:
            predicted_val_naive_bayes_test.append(predict_bayes(j, params))
        predicted_val_naive_bayes_training = []
        for j in training_feat:
            predicted_val_naive_bayes_training.append(predict_bayes(j, params))
        f1_score_test.append(f1_score(labels_test, predicted_val_naive_bayes_test))
        f1_score_training.append(f1_score(training_labels, predicted_val_naive_bayes_training))

        # F1 score for 5% -> 100% of datas noised
        training_labels = add_noise_to_data(labels, percentage[i])
        training_feat = feat
        params = get_distrib_parameters(training_feat, training_labels)
        predicted_val_naive_bayes_test = []
        for j in feat_test:
            predicted_val_naive_bayes_test.append(predict_bayes(j, params))
        predicted_val_naive_bayes_training = []
        for j in training_feat:
            predicted_val_naive_bayes_training.append(predict_bayes(j, params))
        f1_score_noise_test.append(f1_score(labels_test, predicted_val_naive_bayes_test))
        f1_score_noise_training.append(f1_score(training_labels, predicted_val_naive_bayes_training))

        # F1 scores for 10% of datas
        training_feat, training_labels = get_percentage_of_data(feat, labels, percent_10[i])
        params = get_distrib_parameters(training_feat, training_labels)
        predicted_val_naive_bayes_test = []
        for j in feat_test:
            predicted_val_naive_bayes_test.append(predict_bayes(j, params))
        predicted_val_naive_bayes_training = []
        for j in training_feat:
            predicted_val_naive_bayes_training.append(predict_bayes(j, params))
        f1_score_test_10.append(f1_score(labels_test, predicted_val_naive_bayes_test))
        f1_score_training_10.append(f1_score(training_labels, predicted_val_naive_bayes_training))

        # F1 scores for 100% of datas
        training_feat, training_labels = get_percentage_of_data(feat, labels, percent_100[i])
        params = get_distrib_parameters(training_feat, training_labels)
        predicted_val_naive_bayes_test = []
        for j in feat_test:
            predicted_val_naive_bayes_test.append(predict_bayes(j, params))
        predicted_val_naive_bayes_training = []
        for j in training_feat:
            predicted_val_naive_bayes_training.append(predict_bayes(j, params))
        f1_score_test_100.append(f1_score(labels_test, predicted_val_naive_bayes_test))
        f1_score_training_100.append(f1_score(training_labels, predicted_val_naive_bayes_training))

    # Plot figure for percentage of datas from 5% to 100%
    plt.figure()
    plt.title("Overfitting observate with too small volume of datas")
    plt.plot(percentage, f1_score_test, 'o', label="F1 score for test datas")
    plt.plot(percentage, f1_score_training, 'x', label="F1 score for training datas")
    a, b = np.polyfit(np.log(percentage), f1_score_training, 1)
    plt.plot(percentage, a * np.log(percentage) + b, label="Approximation for training datas")
    c, d = np.polyfit(np.log(percentage), f1_score_test, 1)
    plt.plot(percentage, c * np.log(percentage) + d, label="Approximation for test datas")
    plt.xlabel("Percentage of datas used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()
 
    # Plot figure for percentage of datas noised from 5% to 100%
    plt.figure()
    plt.title("Overfitting observate with noise added to datas")
    plt.plot(percentage, f1_score_noise_test, 'o', label="F1 score for test datas")
    plt.plot(percentage, f1_score_noise_training, 'x', label="F1 score for training datas")
    plt.xlabel("Percentage of noise added to datas")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()
 
    # Plot figure for 10% of datas used
    plt.figure()
    plt.title("Overfitting observate with too small volume of datas")
    plt.plot(index, f1_score_test_10, 'o', label="F1 score for test datas")
    plt.plot(index, f1_score_training_10, 'x', label="F1 score for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_training_10), label="Approximation for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_test_10), label="Approximation for test datas")
    plt.xlabel("10% of datas used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()
    
    # Plot figure for 100% of datas used
    plt.figure()
    plt.title("Overfitting observate with too small volume of datas")
    plt.plot(index, f1_score_test_100, 'o', label="F1 score for test datas")
    plt.plot(index, f1_score_training_100, 'x', label="F1 score for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_training_100), label="Approximation for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_test_100), label="Approximation for test datas")
    plt.xlabel("100% of datas used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()



def overfitting_log_reg(feat: np.ndarray, labels: np.ndarray, feat_test: np.ndarray, labels_test: np.ndarray):
    """
    Function to show the overfitting.

    This function will test the performances of the model for a too small set of datas
    """
    if not isinstance(feat, np.ndarray):
        feat = np.asarray(feat)
    if not isinstance(feat_test, np.ndarray):
        feat_test = np.asarray(feat_test)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    if not isinstance(labels_test, np.ndarray):
        labels_test = np.asarray(labels_test)
   
    percentage = np.arange(5, 100)

    percent_10 = [10 for i in range(5,100)]
    percent_100 = [100 for i in range(5,100)]
    index = np.arange(5,100)

    f1_score_test = []
    f1_score_training = []
    f1_score_test_10 = []
    f1_score_training_10 = []
    f1_score_test_100 = []
    f1_score_training_100 = []
    f1_score_noise_test = []
    f1_score_noise_training = []

    for i in range(len(percentage)):

        # F1 score for 5% -> 100% of datas
        training_feat, training_labels = get_percentage_of_data(feat, labels, percentage[i])
        X = np.asarray(training_feat)
        _, n = X.shape
        theta = np.zeros((np.unique(labels).shape[0], n + 1))
        n_it, lr = 1000, 1e-4
        theta = train_log_reg_2(X, training_labels, theta, n_it, lr)
        predicted_val_logreg_test = predict_log_reg_2(feat_test, theta)
        predicted_val_logreg_training = predict_log_reg_2(training_feat, theta)
        f1_score_test.append(f1_score(labels_test, predicted_val_logreg_test))
        f1_score_training.append(f1_score(training_labels, predicted_val_logreg_training))

        # F1 score for 5% -> 100% of datas noised
        training_labels = add_noise_to_data(labels, percentage[i])
        training_feat = feat
        X = np.asarray(feat)
        _, n = X.shape
        theta = np.zeros((np.unique(labels).shape[0], n + 1))
        n_it, lr = 1000, 1e-4
        theta = train_log_reg_2(X, training_labels, theta, n_it, lr)
        predicted_val_logreg_test = predict_log_reg_2(feat_test, theta)
        predicted_val_logreg_training = predict_log_reg_2(training_feat, theta)
        f1_score_noise_test.append(f1_score(labels_test, predicted_val_logreg_test))
        f1_score_noise_training.append(f1_score(training_labels, predicted_val_logreg_training))

        # F1 scores for 10% of datas
        training_feat, training_labels = get_percentage_of_data(feat, labels, percent_10[i])
        X = np.asarray(training_feat)
        _, n = X.shape
        theta = np.zeros((np.unique(labels).shape[0], n + 1))
        n_it, lr = 1000, 1e-4
        theta = train_log_reg_2(X, training_labels, theta, n_it, lr)
        predicted_val_logreg_test = predict_log_reg_2(feat_test, theta)
        predicted_val_logreg_training = predict_log_reg_2(training_feat, theta)
        f1_score_test_10.append(f1_score(labels_test, predicted_val_logreg_test))
        f1_score_training_10.append(f1_score(training_labels, predicted_val_logreg_training))

        # F1 scores for 100% of datas
        training_feat, training_labels = get_percentage_of_data(feat, labels, percent_100[i])
        X = np.asarray(training_feat)
        _, n = X.shape
        theta = np.zeros((np.unique(labels).shape[0], n + 1))
        n_it, lr = 1000, 1e-4
        theta = train_log_reg_2(X, training_labels, theta, n_it, lr)
        predicted_val_logreg_test = predict_log_reg_2(feat_test, theta)
        predicted_val_logreg_training = predict_log_reg_2(training_feat, theta)
        f1_score_test_100.append(f1_score(labels_test, predicted_val_logreg_test))
        f1_score_training_100.append(f1_score(training_labels, predicted_val_logreg_training))

    # Plot figure for percentage of datas from 5% to 100%
    plt.figure()
    plt.title("Overfitting observate with too small volume of datas")
    plt.plot(percentage, f1_score_test, 'o', label="F1 score for test datas")
    plt.plot(percentage, f1_score_training, 'x', label="F1 score for training datas")
    a, b = np.polyfit(np.log(percentage), f1_score_training, 1)
    plt.plot(percentage, a * np.log(percentage) + b, label="Approximation for training datas")
    c, d = np.polyfit(np.log(percentage), f1_score_test, 1)
    plt.plot(percentage, c * np.log(percentage) + d, label="Approximation for test datas")
    plt.xlabel("Percentage of datas used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()
 
    # Plot figure for percentage of datas noised from 5% to 100%
    plt.figure()
    plt.title("Overfitting observate with noise added to datas")
    plt.plot(percentage, f1_score_noise_test, 'o', label="F1 score for test datas")
    plt.plot(percentage, f1_score_noise_training, 'x', label="F1 score for training datas")
    plt.xlabel("Percentage of noise added to datas")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()
 
    # Plot figure for 10% of datas used
    plt.figure()
    plt.title("Overfitting observate with too small volume of datas")
    plt.plot(index, f1_score_test_10, 'o', label="F1 score for test datas")
    plt.plot(index, f1_score_training_10, 'x', label="F1 score for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_training_10), label="Approximation for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_test_10), label="Approximation for test datas")
    plt.xlabel("10% of datas used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()
    
    # Plot figure for 100% of datas used
    plt.figure()
    plt.title("Overfitting observate with too small volume of datas")
    plt.plot(index, f1_score_test_100, 'o', label="F1 score for test datas")
    plt.plot(index, f1_score_training_100, 'x', label="F1 score for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_training_100), label="Approximation for training datas")
    plt.plot(index, 0 * index + np.mean(f1_score_test_100), label="Approximation for test datas")
    plt.xlabel("100% of datas used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()


def overfitting(feat, labels, feat_test, labels_test):

    if not isinstance(feat, np.ndarray):
        feat = np.asarray(feat)
    if not isinstance(feat_test, np.ndarray):
        feat_test = np.asarray(feat_test)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    if not isinstance(labels_test, np.ndarray):
        labels_test = np.asarray(labels_test)

    labels = add_noise_to_data(labels, 50)

    lr = [1e-5]

    n_it = [10**i for i in range(3, 7)]

    f1_score_training = []
    f1_score_test = []

    for j in range(len(lr)):   
        for i in range(len(n_it)):
            theta = np.zeros((np.unique(labels).shape[0], feat.shape[1] + 1))
            theta = train_log_reg_2(feat, labels, theta, n_it[i], lr[j])
            predicted_val_logreg_test = predict_log_reg_2(feat_test, theta)
            predicted_val_logreg_training = predict_log_reg_2(feat, theta)
            f1_score_test.append(f1_score(labels_test, predicted_val_logreg_test))
            f1_score_training.append(f1_score(labels, predicted_val_logreg_training))
    
    plt.figure()
    plt.plot(n_it, f1_score_test, label="F1 score of test dataset")
    plt.plot(n_it, f1_score_training, label="F1 score of training dataset")
    plt.xlabel("Number of iterations")
    plt.xscale("log")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()
 
    plt.figure()
    plt.plot(n_it, np.array(f1_score_test) - np.array(f1_score_training), label="F1 score test - f1 score training")
    plt.xlabel("Number of iterations")
    plt.xscale("log")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()



def main():
    #overfitting_naive_bayes(FEAT, LABELS, FEAT_test, LABELS_test)
    #overfitting_log_reg(FEAT, LABELS, FEAT_test, LABELS_test)
    #overfitting(FEAT, LABELS, FEAT_test, LABELS_test)
    return
