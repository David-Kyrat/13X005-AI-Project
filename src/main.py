import numpy as np
from numpy.typing import NDArray
# import pandas as pd
# import naive_bayes
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo

# Iris dataset
DATASET_ID = 53

iris = fetch_ucirepo(id=DATASET_ID)  # fetch dataset
assert iris.data is not None

DATA: DataFrame = iris.data.original
LAB_NAME: str = iris.data["headers"][-1]

from sklearn.model_selection import train_test_split

FEAT, FEAT_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=np.random.randint(0, 100))
# FEAT, FEAT_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=42)

DATA_train = FEAT.copy(deep=True)
DATA_train["class"] = y_train

DATA_test = FEAT_test.copy(deep=True)
DATA_test["class"] = y_test

LABELS_STR: DataFrame = DATA[LAB_NAME]  # class value as string
# LABELS_STR_test: DataFrame = DATA_test[LAB_NAME]  # type: ignore

lab_values = LABELS_STR.unique()
# lab_values_test = LLAB_IDX_VALABELS_STR_test.unique()

LAB_IDX_VAL: dict[int, str] = dict(zip(range(len(lab_values)), lab_values))  # class index, class value
LAB_VAL_IDX: dict[str, int] = {v: k for k, v in LAB_IDX_VAL.items()}  # class value, class index

LABELS: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_train])
LABELS_test: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_test])
COL_NAMES = list(FEAT.columns)

if __name__ == "__main__":
    # test_gradient_descent()
    # naive_bayes.main()
    # naive_bayes.main()
    pass
