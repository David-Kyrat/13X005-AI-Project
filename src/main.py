import numpy as np
from numpy.typing import NDArray
# import pandas as pd
import naive_bayes
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo

# Iris dataset
DATASET_ID = 53

iris = fetch_ucirepo(id=DATASET_ID)  # fetch dataset
assert iris.data is not None

DATA: DataFrame = iris.data.original

FEAT: DataFrame = iris.data.features
LAB_NAME: str = iris.data["headers"][-1]
LABELS_STR: DataFrame = DATA[LAB_NAME]  # type: ignore
lab_values = LABELS_STR.unique()

LAB_IDX_VAL: dict[int, str] = dict(zip(range(len(lab_values)), lab_values))
LAB_VAL_IDX: dict[str, int] = dict(zip(lab_values, range(len(lab_values))))

LABELS: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in LABELS_STR])
COL_NAMES = list(FEAT.columns)


if __name__ == "__main__":
    # test_gradient_descent()
    naive_bayes.main()
