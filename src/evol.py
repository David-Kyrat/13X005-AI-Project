

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
from overfitting import add_noise_to_data

from main import iris

def evolution_train_volume():
    """
    Visualizes the evolution of F1 scores for Naive Bayes and Logistic Regression 
    models based on varying training data volume.

    Returns:
    - None
    """

    from main import DATA,LAB_NAME,CLASSES,LAB_IDX_VAL,LAB_VAL_IDX,LABELS_STR,iris

    assert iris.data is not None

    # tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=np.random.randint(0, 100))
    tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=27)

    #On fait une liste de différents volume de données d'apprentissage
    vol_train_data=[train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3,train_size=(vol/100), random_state=27) for vol in range(20,71,1)]
    
    train_lengths=[]
    naive_bayes_f1_scores=[]
    log_reg_f1_scores=[]
    naive_bayes_accuracy=[]
    log_reg_accuracy=[]
    naive_bayes_precision=[]
    log_reg_precision=[]
    naive_bayes_recall=[]
    log_reg_recall=[]

    for split_data in vol_train_data:
        tmp_x, tmp_x_test, y_train, y_test=split_data
        train_lengths.append(len(tmp_x)/len(DATA))

        feat, FEAT_test = pd.DataFrame(tmp_x), pd.DataFrame(tmp_x_test)
        feat_test = np.asarray(FEAT_test)

        DATA_train = feat.copy(deep=True)
        DATA_train[LAB_NAME] = y_train

        DATA_test = FEAT_test.copy(deep=True)
        DATA_test[LAB_NAME] = y_test

        labels: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_train])
        LABELS_test: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_test])
        # COL_NAMES = list(FEAT.columns)

        m, n = feat.shape
        init_w = np.random.rand(n)
        init_b = np.random.rand()
        n_it, lr = 1000, 1e-5
        w, b = log_reg.train_log_reg(feat,labels,init_w,init_b,n_it,lr)

        predicted_val_logreg = log_reg.predict_log_reg(FEAT_test.to_numpy(), w, b)
        predicted_nb=[]

        for l in feat_test:
            predicted_nb.append(nb.predict_bayes(l, nb.get_distrib_parameters(feat,labels)))

        #On ajoute à la liste des F1 score, le f1score correspondant au volume de données considéré
        naive_bayes_f1_scores.append(metrics.f1_score(LABELS_test,predicted_nb))
        log_reg_f1_scores.append(metrics.f1_score(LABELS_test,predicted_val_logreg))

        naive_bayes_accuracy.append(metrics.accuracy(LABELS_test,predicted_nb))
        log_reg_accuracy.append(metrics.accuracy(LABELS_test,predicted_val_logreg))

        naive_bayes_precision.append(metrics.precision(LABELS_test,predicted_nb))
        log_reg_precision.append(metrics.precision(LABELS_test,predicted_val_logreg))

        naive_bayes_recall.append(metrics.recall(LABELS_test,predicted_nb))
        log_reg_recall.append(metrics.recall(LABELS_test,predicted_val_logreg))

    #plt.subplot(2,2,1)
    plt.plot(train_lengths,log_reg_f1_scores,"o",label='log_reg')
    plt.plot(train_lengths,naive_bayes_f1_scores,"x",label='naive_bayes')
    plt.legend()
    plt.xlabel("volume de données d'entraînement (en %)")
    plt.ylabel("f1 score (entre 0 et 1)")
    plt.title("Evolution du f1 score en fonction du volume de données d'entraînement")

    plt.show()

    """
    plt.subplot(2,2,2)
    plt.plot(train_lengths,log_reg_accuracy,label='log_reg')
    plt.plot(train_lengths,naive_bayes_accuracy,label='naive_bayes')
    plt.legend()
    plt.xlabel("volume de données d'entraînement (en %)")
    plt.ylabel("Accuracy (entre 0 et 1)")
    plt.title("Evolution de l'accuracy en fonction du volume de données d'entraînement")

    plt.subplot(2,2,3)
    plt.plot(train_lengths,log_reg_precision,label='log_reg')
    plt.plot(train_lengths,naive_bayes_precision,label='naive_bayes')
    plt.legend()
    plt.xlabel("volume de données d'entraînement (en %)")
    plt.ylabel("Precision (entre 0 et 1)")
    plt.title("Evolution de la precision en fonction du volume de données d'entraînement")

    plt.subplot(2,2,4)
    plt.plot(train_lengths,log_reg_recall,label='log_reg')
    plt.plot(train_lengths,naive_bayes_recall,label='naive_bayes')
    plt.legend()
    plt.xlabel("volume de données d'entraînement (en %)")
    plt.ylabel("Recall (entre 0 et 1)")
    plt.title("Evolution du Recall en fonction du volume de données d'entraînement")
    plt.show()
    """


def evolution_noise(train=True):
    """
    Visualizes the evolution of F1 scores for Naive Bayes and Logistic Regression 
    models based on varying levels of Gaussian noise.

    Parameters:
    - train: bool, if True, considers noise on training data; if False, considers noise on test data.

    Returns:
    - None
    """
    
    from main import DATA,LAB_NAME,CLASSES,LAB_IDX_VAL,LAB_VAL_IDX,LABELS_STR,iris

    assert iris.data is not None

    #? Faire le split manuellement afin de pouvoir controller le volume de données d'entrainements et le bruit 

    # tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=np.random.randint(0, 100))
    tmp_x, tmp_x_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=27)

    """"
    noisy_train_data=[add_gaussian_noise(tmp_x,i/10) for i in range(0,51)]
    variances=[i/10 for i in range(0,101)]
    noisy_test_data=[add_gaussian_noise(tmp_x_test,i/10) for i in range(0,51)]
    """
    percentage = np.arange(5, 100)

    naive_bayes_f1_scores=[]
    log_reg_f1_scores=[]
    naive_bayes_accuracy=[]
    log_reg_accuracy=[]
    naive_bayes_precision=[]
    log_reg_precision=[]
    naive_bayes_recall=[]
    log_reg_recall=[]

    for i in range(len(percentage)):

        """"
        if train==True:
            tmp_x=noisy_train_data[i]
        elif train==False:
            tmp_x_test=noisy_test_data[i]
        print("train :",tmp_x)
        """


        feat, FEAT_test = pd.DataFrame(tmp_x), pd.DataFrame(tmp_x_test)

        DATA_train = feat.copy(deep=True)
        DATA_train[LAB_NAME] = y_train

        DATA_test = FEAT_test.copy(deep=True)
        DATA_test[LAB_NAME] = y_test

        labels: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_train])
        labels_test: NDArray = np.array([LAB_VAL_IDX[class_value] for class_value in y_test])
        # COL_NAMES = list(FEAT.columns)

        m, n = feat.shape
        init_w = np.random.rand(n)
        init_b = np.random.rand()
        n_it, lr = 1000, 1e-5

        if train==True:
            labels=add_noise_to_data(labels,percentage[i])
        elif train==False:
            labels_test=add_noise_to_data(labels_test,percentage[i])
    

        w, b = log_reg.train_log_reg(feat,labels,init_w,init_b,n_it,lr)

        predicted_val_logreg = log_reg.predict_log_reg(FEAT_test.to_numpy(), w, b)
        predicted_nb=nb.predict_bayes_all(FEAT_test,params_by_class=nb.get_distrib_parameters(feat,labels))

        #On ajoute à la liste des F1 score, le f1score correspondant au volume de données considéré
        naive_bayes_f1_scores.append(metrics.f1_score(labels_test,predicted_nb))
        log_reg_f1_scores.append(metrics.f1_score(labels_test,predicted_val_logreg))

        naive_bayes_accuracy.append(metrics.accuracy(labels_test,predicted_nb))
        log_reg_accuracy.append(metrics.accuracy(labels_test,predicted_val_logreg))

        naive_bayes_precision.append(metrics.precision(labels_test,predicted_nb))
        log_reg_precision.append(metrics.precision(labels_test,predicted_val_logreg))

        naive_bayes_recall.append(metrics.recall(labels_test,predicted_nb))
        log_reg_recall.append(metrics.recall(labels_test,predicted_val_logreg))

    #plt.subplot(1,2,1)
    plt.plot(percentage,log_reg_f1_scores,"o",label='log_reg')
    plt.plot(percentage,naive_bayes_f1_scores,"x",label='naive_bayes')

    a, b = np.polyfit(percentage, log_reg_f1_scores, 1)
    plt.plot(percentage, a * percentage + b, label="Approximation for logistic regression predictions")

    c, d = np.polyfit(percentage, naive_bayes_f1_scores, 1)
    plt.plot(percentage, c * percentage + d, label="Approximation for naive bayes predictions")

    plt.legend()
    plt.xlabel("variance du bruit")
    plt.ylabel("f1 score (entre 0 et 1)")
    if train==True:
            plt.title("Evolution du f1 score en fonction du bruit sur les données d'entraînement")
    elif train==False:
        plt.title("Evolution du f1 score en fonction du bruit sur les données de test")

    plt.show()

    """
    plt.subplot(1,2,2)
    plt.plot(variances,log_reg_accuracy,label='log_reg')
    plt.plot(variances,naive_bayes_accuracy,label='naive_bayes')
    plt.legend()
    plt.xlabel("variance du bruit")
    plt.ylabel("Accuracy (entre 0 et 1)")
    if train==True:
            plt.title("Evolution de l'accuracy en fonction du bruit sur les données d'entraînement")
    elif train==False:
        plt.title("Evolution de l'accuracy en fonction du bruit sur les données de test")
    plt.show()
    """


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
