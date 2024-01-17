#! CE CODE NE MARCHE PAS TANT QUE NOUS NE POUVONS PAS PASSER LES DONEES D'APPRENTISSAGE EN PARAMETRE 

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# import naive_bayes
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo
import metrics
import naive_bayes as nb
import matplotlib.pyplot as plt

import naive_bayes, log_reg,metrics

# Iris dataset
DATASET_ID = 53

iris = fetch_ucirepo(id=DATASET_ID)  # fetch dataset
assert iris.data is not None

DATA: DataFrame = iris.data.original
LAB_NAME: str = iris.data["headers"][-1]

#? Faire le split manuellement afin de pouvoir controller le volume de données d'entrainements et le bruit ?
from sklearn.model_selection import train_test_split

# tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=np.random.randint(0, 100))
tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=27)

#On fait une liste de différents volume de données d'apprentissage
train_data=[train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3,train_size=(vol/10), random_state=27) for vol in range(2,8,1)]
train_lengths=[]
f1_scores=[]

for split_data in train_data:
    tmp_x, tmp_x_test, y_train, y_test=split_data
    train_lengths.append(len(tmp_x))

    FEAT, FEAT_test = pd.DataFrame(tmp_x), pd.DataFrame(tmp_x_test)

    DATA_train = FEAT.copy(deep=True)
    DATA_train[LAB_NAME] = y_train

    DATA_test = FEAT_test.copy(deep=True)
    DATA_test[LAB_NAME] = y_test

    LABELS_STR = DATA[LAB_NAME]  # class value as string
    # LABELS_STR_train = y_train  # non-serialized class value of test dataset
    # LABELS_STR_test: DataFrame = y_test  # type: ignore

    CLASSES = LABELS_STR.unique()

    # seriliazation of class value from string to int (we just take the indices)
    LAB_IDX_VAL: dict[int, str] = dict(zip(range(len(CLASSES)), CLASSES))
    LAB_VAL_IDX: dict[str, int] = dict(zip(CLASSES, range(len(CLASSES))))

    LABELS: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_train])
    LABELS_test: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_test])
    # COL_NAMES = list(FEAT.columns)

    w, b = np.array([0.53452349, 0.36463584, 1.16132476, 1.08204578]), 0.45146791

    predicted_val_logreg = log_reg.predict_log_reg(FEAT_test.to_numpy(), w, b)

    #On ajoute à la liste des F1 score, le f1score correspondant au volume de données considéré
    f1_scores.append(metrics.f1_score(LABELS_test,nb.predict_bayes_all(FEAT_test)))
    #f1_scores.append(metrics.f1_score(LABELS_test,predicted_val_logreg))

if __name__ == "__main__":  # noqa: F401

    # test_gradient_descent()
    # naive_bayes.main()
    #naive_bayes.main()
    print(f1_scores)
    print(train_lengths)
    plt.plot(train_lengths,f1_scores)
    plt.show()
    #metrics.test_metrics()
    # log_reg.main()
    # pass
