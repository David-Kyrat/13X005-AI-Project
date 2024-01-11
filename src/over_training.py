import numpy as np
import matplotlib.pyplot as plt
from log_reg import train_log_reg, predict_log_reg, predict_compute_metrics
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
    return newfeat, newlabels

def over_training_log_reg(feat: np.ndarray, labels: np.ndarray, feat_test: np.ndarray, labels_test: np.ndarray):
    """
    Function to show the over training.

    The over training can be caused by redondant caracteristic, by a too small set of datas, by a too complex model or by too much interaction between caracteristics.

    This function will test the performances of the model for a too small set of datas and for redondant caracteristic
    """
    if not isinstance(feat, np.ndarray):
        feat = np.asarray(feat)
    if not isinstance(feat_test, np.ndarray):
        feat_test = np.asarray(feat_test)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    if not isinstance(labels_test, np.ndarray):
        labels_test = np.asarray(labels_test)
   
    percentage = [10 * i for i in range(1, 11)]

    f1_score_test = []
    f1_score_training = []

    for i in range(len(percentage)):
        training_feat, training_labels = get_percentage_of_data(feat, labels, percentage[i])
        X = np.asarray(training_feat)
        _, n = X.shape
        init_w = np.random.rand(n)
        init_b = np.random.rand()
        n_it, lr = 1000, 1e-5
        w, b = train_log_reg(X, training_labels, init_w, init_b, n_it, lr)
        predicted_val_logreg_test = predict_log_reg(feat_test, w, b)
        predicted_val_logreg_training = predict_log_reg(training_feat, w, b)
        f1_score_test.append(f1_score(labels_test, predicted_val_logreg_test))
        f1_score_training.append(f1_score(training_labels, predicted_val_logreg_training))

    plt.figure()
    plt.title("Over training observate with too small volume of datas")
    plt.plot(percentage, f1_score_test, label="F1 score for test datas")
    plt.plot(percentage, f1_score_training, label="F1 score for training datas")
    plt.xlabel("Percentage of datas used")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()

    # Add redondant parameters

    f1_score_test_2 = []
    f1_score_training_2 = []

    number_of_redondant_parameters = 10
    training_feat = feat.copy()
    test_feat = feat_test.copy()
    redondance_list = np.arange(number_of_redondant_parameters)
    for i in range(number_of_redondant_parameters):
        training_feat = np.append(training_feat, np.ones((len(training_feat),1)) * i, axis = 1)
        test_feat = np.append(test_feat, np.ones((len(test_feat),1)) * i, axis = 1)
        X = np.asarray(training_feat)
        _, n = X.shape
        init_w = np.random.rand(n)
        init_b = np.random.rand()
        n_it, lr = 1000, 1e-5
        w, b = train_log_reg(X, labels, init_w, init_b, n_it, lr)
        predicted_val_logreg_test = predict_log_reg(test_feat, w, b)
        predicted_val_logreg_training = predict_log_reg(training_feat, w, b)
        f1_score_test_2.append(f1_score(labels_test, predicted_val_logreg_test))
        f1_score_training_2.append(f1_score(labels, predicted_val_logreg_training))


    plt.figure()
    plt.title("Over training observate with redondant caracteristics")
    plt.plot(redondance_list, f1_score_test_2, label="F1 score for test datas")
    plt.plot(redondance_list, f1_score_training_2, label="F1 score for training datas")
    plt.xlabel("Number of redondant caracteristics")
    plt.ylabel("f1 score")
    plt.legend()
    plt.show()



over_training_log_reg(FEAT, LABELS, FEAT_test, LABELS_test)







