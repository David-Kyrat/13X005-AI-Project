import numpy as np
from numpy.typing import NDArray
import pandas as pd

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

# tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=np.random.randint(0, 100))
tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=27)

FEAT, FEAT_test = pd.DataFrame(tmp_x), pd.DataFrame(tmp_x_test)
DATA_train = FEAT.copy(deep=True)
DATA_train[LAB_NAME] = y_train

DATA_test = FEAT_test.copy(deep=True)
DATA_test[LAB_NAME] = y_test

LABELS_STR = DATA[LAB_NAME]  # class value as string
LABELS_STR_train = y_train  # non-serialized class value of test dataset
LABELS_STR_test: DataFrame = y_test  # type: ignore

CLASSES = LABELS_STR.unique()

LAB_IDX_VAL: dict[int, str] = dict(zip(range(len(CLASSES)), CLASSES))
LAB_VAL_IDX: dict[str, int] = dict(zip(CLASSES, range(len(CLASSES))))

LABELS: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_train])
LABELS_test: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_test])
# COL_NAMES = list(FEAT.columns)


if __name__ == "__main__":
    import naive_bayes, log_reg  # noqa: F401

    # test_gradient_descent()
    # naive_bayes.main()
    naive_bayes.main()
    # log_reg.main()
    # pass
