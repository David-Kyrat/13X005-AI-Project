# import numpy as np
# import pandas as pd
import naive_bayes
from gradient_descent import gradient_descent, test_gradient_descent
from naive_bayes import *  # noqa: F403, F401
from pandas import DataFrame, Series
from ucimlrepo import fetch_ucirepo

# Iris dataset
DATASET_ID = 53

iris = fetch_ucirepo(id=DATASET_ID)  # fetch dataset
assert iris.data is not None

DATA: DataFrame = iris.data.original

FEAT: DataFrame = iris.data.features
LAB_NAME: str = iris.data["headers"][-1]
LABELS = DATA[LAB_NAME]
COL_NAMES = list(FEAT.columns)


if __name__ == "__main__":
    print(LAB_NAME)
    exit(0)
    # pass
    # test_gradient_descent()
    naive_bayes.main()
