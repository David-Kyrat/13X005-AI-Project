
#! CE CODE NE MARCHE PAS TANT QUE NOUS NE POUVONS PAS PASSER LES DONEES D'APPRENTISSAGE EN PARAMETRE 
#TODO : Ajouter des commentaires + formaliser l'explication des fonctions

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo
import metrics
import naive_bayes as nb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import naive_bayes, log_reg,metrics

def add_gaussian_noise_to_dataframe(df, mean=0, std=1):
    noise = pd.DataFrame(np.random.normal(mean, std, df.shape), columns=df.columns)
    noisy_df = df + noise
    return noisy_df

def evolution_train_volume():
    from main import DATA,LAB_NAME,CLASSES,LAB_IDX_VAL,LAB_VAL_IDX,LABELS_STR,iris

    assert iris.data is not None

    #? Faire le split manuellement afin de pouvoir controller le volume de données d'entrainements et le bruit 

    # tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=np.random.randint(0, 100))
    tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=27)

    #On fait une liste de différents volume de données d'apprentissage
    vol_train_data=[train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3,train_size=(vol/10), random_state=27) for vol in range(2,8,1)]
    
    train_lengths=[]
    naive_bayes_f1_scores=[]
    log_reg_f1_scores=[]

    for split_data in vol_train_data:
        tmp_x, tmp_x_test, y_train, y_test=split_data
        train_lengths.append(len(tmp_x))

        FEAT, FEAT_test = pd.DataFrame(tmp_x), pd.DataFrame(tmp_x_test)

        DATA_train = FEAT.copy(deep=True)
        DATA_train[LAB_NAME] = y_train

        DATA_test = FEAT_test.copy(deep=True)
        DATA_test[LAB_NAME] = y_test

        LABELS: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_train])
        LABELS_test: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_test])
        # COL_NAMES = list(FEAT.columns)

        w, b = np.array([0.53452349, 0.36463584, 1.16132476, 1.08204578]), 0.45146791

        predicted_val_logreg = log_reg.predict_log_reg(FEAT_test.to_numpy(), w, b)

        #On ajoute à la liste des F1 score, le f1score correspondant au volume de données considéré
        log_reg_f1_scores.append(metrics.f1_score(LABELS_test,nb.predict_bayes_all(FEAT_test)))
        naive_bayes_f1_scores.append(metrics.f1_score(LABELS_test,predicted_val_logreg))

    print("nb f1scores",naive_bayes_f1_scores)
    print("logreg f1scores",log_reg_f1_scores)
    print(train_lengths)
    plt.plot(train_lengths,log_reg_f1_scores,label='log_reg')
    plt.plot(train_lengths,naive_bayes_f1_scores,label='naive_bayes')
    plt.legend()
    plt.xlabel("volume de données d'entraînement (en %)")
    plt.ylabel("f1 score (entre 0 et 1)")
    plt.title("Evolution du f1 score en fonction du volume de données d'entraînement")
    plt.show()


def evolution_noise(train=True):

    from main import DATA,LAB_NAME,CLASSES,LAB_IDX_VAL,LAB_VAL_IDX,LABELS_STR,iris

    assert iris.data is not None

    #? Faire le split manuellement afin de pouvoir controller le volume de données d'entrainements et le bruit 

    # tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=np.random.randint(0, 100))
    tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=27)


    noisy_train_data=[add_gaussian_noise(tmp_x,std=1/i) for i in range(10,0,-1)]
    variances=[1/i for i in range(10,0,-1)]
    noisy_test_data=[add_gaussian_noise(tmp_x_test,std=1/i) for i in range(10,0,-1)]
    
    naive_bayes_f1_scores=[]
    log_reg_f1_scores=[]

    for i in range(len(noisy_train_data)):
        if train==True:
            tmp_x=noisy_train_data[i]
        elif train==False:
            tmp_x_test=noisy_test_data[i]


        FEAT, FEAT_test = pd.DataFrame(tmp_x), pd.DataFrame(tmp_x_test)

        DATA_train = FEAT.copy(deep=True)
        DATA_train[LAB_NAME] = y_train

        DATA_test = FEAT_test.copy(deep=True)
        DATA_test[LAB_NAME] = y_test

        LABELS: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_train])
        LABELS_test: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_test])
        # COL_NAMES = list(FEAT.columns)

        w, b = np.array([0.53452349, 0.36463584, 1.16132476, 1.08204578]), 0.45146791

        predicted_val_logreg = log_reg.predict_log_reg(FEAT_test.to_numpy(), w, b)

        #On ajoute à la liste des F1 score, le f1score correspondant au volume de données considéré
        log_reg_f1_scores.append(metrics.f1_score(LABELS_test,nb.predict_bayes_all(FEAT_test)))
        naive_bayes_f1_scores.append(metrics.f1_score(LABELS_test,predicted_val_logreg))
    
    print("nb f1scores",naive_bayes_f1_scores)
    print("logreg f1scores",log_reg_f1_scores)
    print(train_lengths)
    plt.plot(variances,log_reg_f1_scores,label='log_reg')
    plt.plot(variances,naive_bayes_f1_scores,label='naive_bayes')
    plt.legend()
    plt.xlabel("variance du bruit")
    plt.ylabel("f1 score (entre 0 et 1)")
    if train==True:
            plt.title("Evolution du f1 score en fonction du bruit sur les données d'entraînement")
    elif train==False:
        plt.title("Evolution du f1 score en fonction du bruit sur les données de test")
    plt.show()


if __name__ == "__main__":  # noqa: F401

    # test_gradient_descent()
    # naive_bayes.main()
    #naive_bayes.main()
    evolution_train_volume()
    evolution_noise()
    evolution_noise(train=False)
    #metrics.test_metrics()
    # log_reg.main()
    # pass
