from pprint import pprint
from typing import Any

import numpy as np

# NB: floating is any (numpy) floating type NDArray or not
from numpy import float32 as f32, floating as fl

from numpy.typing import NDArray
from pandas import DataFrame


def normal_pdf(mean: fl, std: fl):
    """
    Parameters
    ----------
    `mean` : float or NDArray of float
        The mean (μ) of the normal distribution.
    `std : float or NDArray of float
        The standard deviation (σ) of the normal distribution.
    Returns
    -------
    A lambda function representing the normal distribution's PDF,
    i.e.  (1 / (σ * sqrt(2π))) * exp(-((x - μ)² / (2σ²)))."""
    return lambda x: (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std**2))


def get_distrib_parameters(data: DataFrame, labels: DataFrame) -> dict[Any, list[tuple[fl, fl]]]:
    """
    Parameters
    ----------
    `data` : The dataset.
    `labels` : Labels to extract the different values from (will be the keys of the return dict)
    Returns
    -------
    Parameters for each distribution of each feature feature for each class.
    i.e. a dictionary {class: [(mean_i, std_i), ...]} for each feature i."""
    from main import lab_values as classes

    # classes = labels.unique()
    out: dict[Any, list[tuple[fl, fl]]] = {}
    for classv in classes:
        out_classv = []  # list of (mean, std) for each feature by class value
        data_c = data[labels == classv]  # data for current class
        for feature in data_c:
            feat = data_c[feature]
            mean, std = feat.mean(), feat.std()
            out_classv.append((f32(mean), f32(std)))
        out[classv] = out_classv

    return out


def predict_bayes(x, params_by_class: dict[Any, list[tuple[fl, fl]]]) -> Any:
    """
    Parameters
    ----------
    `x` : NDArray or list or tuple
        The (*unique*) sample to predict.
    `params_by_class` : The parameters of the normal distribution of each feature for each class.
    Returns
    -------
    The predicted class for the sample x."""
    # if type(x) is not np.ndarray: x = np.asarray(x)
    probs = {}
    for class_value, params in params_by_class.items():
        # Computes \Prod{ P(X_i | y) } for all i and for current y = class_value
        probs[class_value] = np.prod([normal_pdf(mean, std)(x_i) for (mean, std), x_i in zip(params, x)])
    # get the class that maximize the conditional probability
    return max(probs, key=lambda class_value: probs[class_value])

#
# def predict_bayes_all(X) -> NDArray:
#     """
#     Parameters
#     ----------
#     `X` : NDArray, Feature matrix
#         All samples to predict.
#     `params_by_class` : The parameters of the normal distribution of each feature for each class.
#     Returns
#     -------
#     The predicted classes for each sample in X."""
#     from concurrent.futures import ThreadPoolExecutor
#
#     MAX_WORKERS = 2**8
#     executor = ThreadPoolExecutor(MAX_WORKERS)
#     futures = []
#     for x in X:
#         futures.append(executor.submit(predict_bayes, x, None))
#     return np.array([f.result() for f in futures])

# ================================================================
# ======================= TEST:==================================
# ================================================================


def test_get_normal_parameters():
    from main import FEAT, COL_NAMES, LABELS_STR

    params_by_class = get_distrib_parameters(FEAT, COL_NAMES, LABELS_STR)  # type: ignore
    print("Format: (mean_i, std_i), ...,  for each class")
    pprint(params_by_class)


def test_predict_bayes_runs():
    from main import FEAT, LABELS_STR

    params_by_class = get_distrib_parameters(FEAT, LABELS_STR)  # type: ignore
    # test sample
    idx = np.random.randint(0, len(FEAT))
    x = FEAT.iloc[idx]  # type: ignore
    print("Sample to predict:\n", x, "\n ")
    pred = predict_bayes(x, params_by_class)
    print("Predicted class: ", pred)
    print("Actual class: ", LABELS_STR.iloc[idx])


def test_predict_bayes_f1score():
    from main import FEAT, FEAT_test, LABELS_test, DATA_train, LAB_NAME, LAB_IDX_VAL, LABELS
    import pandas as pd  # noqa: F401

    correct_classes = DATA_train[LAB_NAME]
    params_by_class = get_distrib_parameters(FEAT, correct_classes)  # type: ignore
    # params_by_class = get_distrib_parameters(FEAT, COL_NAMES, pd.Series(LABELS))  # type: ignore
    print("-------------------------------")
    # test sample
    mistake = 0
    # for i, sample in enumerate(FEAT_test.itertuples()):
    for sample, correct_class in zip(FEAT_test.itertuples(), correct_classes):
        pred = predict_bayes(sample, params_by_class)
        print("Predicted class: ", pred)
        print("Actual class: ", correct_class)
        if pred != correct_class:
            mistake += 1
        print()
    precision = 1 - (mistake / len(FEAT_test))
    print("Precision: ", len(FEAT_test) - mistake, f"/ {len(FEAT_test)}", " (", precision * 100, "%)")
    # idx = np.random.randint(0, len(FEAT))
    # x = FEAT.iloc[idx]  # type: ignore
    # print("Sample to predict:\n", x, "\n ")
    # pred = predict_bayes(x, params_by_class)


def main():
    # test_get_normal_parameters()
    print(" ")
    test_predict_bayes_f1score()
