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


def get_distrib_parameters(data: DataFrame, feature_names: list[str], labels: DataFrame) -> dict[Any, list[tuple[fl, fl]]]:
    """
    Parameters
    ----------
    `data` : The dataset.
    `feature_names` : The names of the features to extract the normal parameters from.
    `labels` : Labels to extract the different values from (will be the keys of the return dict)
    Returns
    -------
    Parameters for each distribution of each feature feature for each class.
    i.e. a dictionary {class: [(mean_i, std_i), ...]} for each feature i."""
    classes = labels.unique()
    out: dict[Any, list[tuple[fl, fl]]] = {}
    for classv in classes:
        out_classv = []  # list of (mean, std) for each feature by class value
        data_c = data[labels == classv]  # data for current class
        for feature in feature_names:
            feat = data_c[feature]
            mean, std = feat.mean(), feat.std()
            out_classv.append((f32(mean), f32(std)))
        out[classv] = out_classv

    return out


def predict_bayes(x: NDArray, params_by_class: dict[Any, list[tuple[fl, fl]]]) -> Any:
    """
    Parameters
    ----------
    `x` : The sample to predict.
    `params_by_class` : The parameters of the normal distribution of each feature for each class.
    Returns
    -------
    The predicted class for the sample x."""
    probs = {}
    if type(x) is not np.ndarray:
        x = np.asarray(x)

    for class_value, params in params_by_class.items():
        probs[class_value] = 1
        for feature_idx, (mean, std) in enumerate(params):
            x_i = x[feature_idx]
            probs[class_value] *= normal_pdf(mean, std)(x_i)  # computes P(X_i | y) for current y = class_value
    # get the class that maximize the conditional probability
    return max(probs, key=lambda class_value: probs[class_value])


# ================================================================
# ======================= TEST:==================================
# ================================================================


def test_get_normal_parameters():
    from main import FEAT, COL_NAMES, LABELS_STR
    params_by_class = get_distrib_parameters(FEAT, COL_NAMES, LABELS_STR)
    print("Format: (mean_i, std_i), ...,  for each class")
    pprint(params_by_class)


def test_predict_bayes():
    from main import FEAT, COL_NAMES, LABELS_STR
    params_by_class = get_distrib_parameters(FEAT, COL_NAMES, LABELS_STR)
    # test sample
    idx = np.random.randint(0, len(FEAT))
    x = FEAT.iloc[idx]
    print("Sample to predict:\n", x, "\n ")
    pred = predict_bayes(x, params_by_class)
    print("Predicted class: ", pred)
    print("Actual class: ", LABELS_STR.iloc[idx])


def main():
    # test_get_normal_parameters()
    print(" ")
    test_predict_bayes()
