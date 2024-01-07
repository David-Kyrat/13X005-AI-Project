from pprint import pprint
from typing import Any

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from numpy import int32 as i32  # noqa: F401
from numpy import float64 as f64, float32 as f32, floating as fl

import pandas as pd  # noqa: F401
import plot_util  # noqa: F401
from main import DATASET_ID, FEAT, LABELS, COL_NAMES  # noqa: F401
from numpy.typing import NDArray
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo


def test_ucml_repo():
    iris = fetch_ucirepo(id=DATASET_ID)  # fetch dataset
    assert iris.data is not None

    data: DataFrame = iris.data.original

    _X: DataFrame = iris.data.features
    _y: DataFrame = iris.data.targets

    pprint(data)
    # pprint(_X)
    # pprint(_y)


def normal_pdf(mean: fl, std: fl):
    """
    Parameters
    ----------
    mean : float or NDArray of float
        The mean (μ) of the normal distribution.
    std : float or NDArray of float
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
    data : The dataset.
    feature_names : The names of the features to extract the normal parameters from.
    labels : Labels to extract the different values from (will be the keys of the return dict)
    Returns
    -------
    Parameters for each distribution of each feature feature for each class.
    i.e. a dictionary {class: [(mean_i, std_i), ...]} for each feature i."""
    classes = labels.unique()
    out: dict[Any, list[tuple[fl, fl]]] = {}
    for classv in classes:
        out[classv] = []
        # data for current class
        data_c = data[labels == classv]
        for feature in feature_names:
            feat = data_c[feature]
            mean, std = feat.mean(), feat.std()
            out[classv].append((f32(mean), f32(std)))

    return out


def predict_bayes(x: NDArray, params_by_class: dict[Any, list[tuple[fl, fl]]]) -> Any:
    """
    Parameters
    ----------
    x : The sample to predict.
    params_by_class : The parameters of the normal distribution of each feature for each class.
    Returns
    -------
    The predicted class for the sample x."""
    probs = {}
    for class_value, params in params_by_class.items():
        probs[class_value] = 1
        for feature_idx, (mean, std) in enumerate(params):
            x_i = x[feature_idx]
            probs[class_value] *= normal_pdf(mean, std)(x_i)  # computes P(X_i | y) for current y = class_value
    # get the class that maximize the conditional probability
    return max(probs, key=lambda class_value: probs[class_value])


def test_get_normal_parameters():
    params_by_class = get_distrib_parameters(FEAT, COL_NAMES, LABELS)
    print("Format: (mean_i, std_i), ...,  for each class")
    pprint(params_by_class)

def test_predict_bayes():
    params_by_class = get_distrib_parameters(FEAT, COL_NAMES, LABELS)
    # test sample
    idx = np.random.randint(0, len(FEAT))
    x = FEAT.iloc[idx]
    print("Sample to predict:\n", x, "\n ")
    pred = predict_bayes(x, params_by_class)
    print("Predicted class: ", pred)
    print("Actual class: ", LABELS.iloc[idx])


def main():
    # test_get_normal_parameters()
    print(" ")
    test_predict_bayes()
