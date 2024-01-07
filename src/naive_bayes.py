from pprint import pprint

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import plot_util  # noqa: F401
from main import DATASET_ID
from numpy.typing import NDArray  # noqa: F401
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
