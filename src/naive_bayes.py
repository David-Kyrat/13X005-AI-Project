from pprint import pprint
from typing import Any

import numpy as np
import metrics

# NB: floating is any (numpy) floating type NDArray or not
from numpy import float32 as f32
from numpy import floating as fl
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


def get_distrib_parameters(features: DataFrame, labels) -> dict[Any, list[tuple[fl, fl]]]:
    """
    Parameters
    ----------
    `features` : Features from training dataset (only the features, i.e. `main.FEAT`).
    `labels` : `Series` or `DataFrame` (i.e. a column, `main.LABELS`)
        Labels (from training dataset) to extract the different values from (will be the keys of the returned dict)
    Returns
    -------
    Parameters for each distribution of each feature feature for each class.
    i.e. a dictionary {class: [(mean_i, std_i), ...]} for each feature i."""
    from main import LAB_IDX_VAL

    out: dict[Any, list[tuple[fl, fl]]] = {}
    for classv in LAB_IDX_VAL.keys():
        out_classv = []  # list of (mean, std) for each feature by class value
        data_c = features[labels == classv]  # data for current class
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


def predict_bayes_all(X: DataFrame, params_by_class: dict[Any, list[tuple[fl, fl]]] | None = None) -> list[Any]:
    """Computes concurrently predictions for all samples in X. (1 thread per sample)
    Parameters
    ----------
    `X` : Features / All samples to predict (`main.FEAT`).
    `params_by_class` : The parameters of the normal distribution of each feature for each class.
    Returns
    -------
    The predicted classes for each sample in X."""
    from main import FEAT, LABELS

    if params_by_class is None:
        params_by_class = get_distrib_parameters(FEAT, LABELS)

    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(len(X))
    futures = [executor.submit(predict_bayes, x[1:], params_by_class) for x in X.itertuples()]
    return [f.result() for f in futures]


# ================================================================
# ======================= TEST:===================================
# ================================================================


def test_get_normal_parameters():
    from main import FEAT, LABELS

    params_by_class = get_distrib_parameters(FEAT, LABELS)  # type: ignore
    print("Format: (mean_i, std_i), ...,  for each class")
    pprint(params_by_class)


def test_predict_bayes_runs():
    from main import FEAT, LABELS

    params_by_class = get_distrib_parameters(FEAT, LABELS)  # type: ignore
    idx = np.random.randint(0, len(FEAT))  # test sample
    x = FEAT.iloc[idx]  # type: ignore
    print("Sample to predict:\n", x, "\n ")
    _ = predict_bayes(x, params_by_class)  # just tests that it runs


def test_predict_bayes_seq_score():
    from main import FEAT, FEAT_test, LABELS_test, LABELS  # noqa: F401

    params_by_class = get_distrib_parameters(FEAT, LABELS)
    predicted = [predict_bayes(sample[1:], params_by_class) for sample in FEAT_test.itertuples()]
    scores = metrics.compute_metrics(LABELS_test, predicted)
    pprint(scores)


def test_predict_bayes_f1score_all():
    from main import FEAT_test, LABELS_test

    predicted = predict_bayes_all(FEAT_test)
    scores = metrics.compute_metrics(LABELS_test, predicted)
    print()
    pprint(scores)


def main():
    # test_get_normal_parameters()
    print(" ")
    test_predict_bayes_f1score_all()
