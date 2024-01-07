# import numpy as np
# import pandas as pd
import naive_bayes
from gradient_descent import gradient_descent, test_gradient_descent
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo

# Iris dataset
DATASET_ID = 53

iris = fetch_ucirepo(id=DATASET_ID)  # fetch dataset
assert iris.data is not None

DATA: DataFrame = iris.data.original

FEAT: DataFrame = iris.data.features
LAB_NAME: str = iris.data["headers"][-1]
LABELS: DataFrame = DATA[LAB_NAME]  # type: ignore
COL_NAMES = list(FEAT.columns)


if __name__ == "__main__":
    # test_gradient_descent()
    naive_bayes.main()

