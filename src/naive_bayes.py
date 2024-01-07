from pprint import pprint
from typing import Any

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from numpy import int32 as i32  # noqa: F401
from numpy import float64 as f64, floating as fl

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


def normal_pdf(mean: NDArray, std: NDArray):
    """
    Parameters
    ----------
    mean : NDArray, (mostly a float, i.e. N=0)
        The mean (μ) of the normal distribution.
    std : NDArray, (mostly a float, i.e. N=0)
        The standard deviation (σ) of the normal distribution.
    Returns
    -------
    A lambda function representing the normal distribution's PDF,
    i.e.  (1 / (σ * sqrt(2π))) * exp(-((x - μ)² / (2σ²)))."""
    return lambda x: (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std**2))


def get_normal_parameters(data: DataFrame, feature_names: list[str], labels: DataFrame) -> dict[Any, list[tuple[fl, fl]]]:
    """
    Parameters
    ----------
    data : The dataset.
    feature_names : The names of the features to extract the normal parameters from.
    labels : Labels to extract the different values from (will be the keys of the return dict)
    Returns
    -------
    A dictionary {class: [(mean_i, std_i), ...]} for each feature i in the dataset."""

    classes = labels.unique()
    out: dict[Any, list[tuple[fl, fl]]] = {}
    for classv in classes:
        out[classv] = []
        # data for current class
        data_c = data[labels == classv]
        for feature in feature_names:
            feat = data_c[feature]
            mean, std = feat.mean(), feat.std()
            out[classv].append((f64(mean), f64(std)))

    return out


def main():
    _a = get_normal_parameters(FEAT, COL_NAMES, LABELS)
