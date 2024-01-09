
#TODO : COMMENT
#TODO : Adapt to the logistic regression implementation (columns names...)

def evaluation(true_labels,predict_labels,label):
    """
    This function evaluates a prediction by calculating the true/false positive/negative, which are necessary for the below metrics.

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels
    
    Return value
    ------------
    tuple of int (4 elements wich are the true/false positive/negative values)
    """
    true_negative=sum(real == label and pred!=label for real, pred in zip(true_labels, predict_labels))
    true_positive=sum(real == label and pred== label for real, pred in zip(true_labels, predict_labels))
    false_negative=sum(real == label and pred!= label for real, pred in zip(true_labels, predict_labels))
    false_positive=sum(true != label and pred== label for true, pred in zip(true_labels, predict_labels))

    return true_positive,false_positive,true_negative,false_negative


def accuracy(true_labels, predicted_labels):
    """
    This function implements the accuracy metric wich measure the proportion of correct predictions considering all predictions

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels
    
    Return value
    ------------
    float number wich is the accuracy proportion value between 0 and 1.
    """

    correct_predictions = sum(true == pred for true, pred in zip(true_labels, predicted_labels))
    total_instances = len(true_labels)
    return correct_predictions / total_instances


def precision(true_labels,predict_labels):
    """
    This function implements the precision metric wich measure the proportion of false positive predictions considering every positive predictions for each label, and returns the average precision

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels
    
    Return value
    ------------
    float number wich is the precision value between 0 and 1.
    """
    
    labels_names=list(set(true_labels))
    precisions=[]
    for label in labels_names:
        true_positive, false_positive, _, _=evaluation(true_labels,predict_labels,label)
        precisions.append(true_positive / (true_positive + false_positive))
    
    return sum(precisions)/len(labels_names)


def recall(true_labels,predict_labels):
    """
    This function implements the precision metric wich measure the proportion of uncorrect negatives predictions (recall of 1 means the number of fale negatives is null) for each label, and returns the average recall

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels
    
    Return value
    ------------
    float number wich is the recall value between 0 and 1.
    """

    labels_names=list(set(true_labels))
    recalls=[]
    for label in labels_names:
        true_positive, _, _, false_negative=evaluation(true_labels,predict_labels,label)
        recalls.append(true_positive / (true_positive + false_negative))
    
    return sum(recalls)/len(labels_names)


def f1_score(true_labels,predict_labels):
    """
    This function implements the f1 score metric wich is a balanced measure between precision and recall that measures errors more broadly

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: 1d array-like
        predicted labels
    
    Return value
    ------------
    float number wich is the f1 score value between 0 and 1.
    """

    prec=precision(true_labels,predict_labels)
    rec=recall(true_labels,predict_labels)
    return 2 * (prec * rec) / (prec + rec)



# ================================================================
# ======================= TEST:==================================
# ================================================================

#Extracting data :

from main import DATA, LAB_NAME, iris, COL_NAMES
from sklearn.model_selection import train_test_split
from pandas import DataFrame

FEAT, FEAT_test, y_train, y_test = train_test_split(iris.data.features, DATA[LAB_NAME], test_size=0.3, random_state=42)

DATA_train = FEAT.copy(deep=True)
DATA_train["class"] = y_train

DATA_test = FEAT_test.copy(deep=True)
DATA_test["class"] = y_test

LABELS_STR: DataFrame = DATA_train[LAB_NAME]  # type: ignore
LABELS_STR_test: DataFrame = DATA_test[LAB_NAME]  # type: ignore

from naive_bayes import predict_bayes, get_distrib_parameters
import sklearn.metrics


#TRAIN :

parameters=get_distrib_parameters(DATA_train,COL_NAMES,LABELS_STR)

predicitons=[predict_bayes(FEAT.iloc[idx], parameters) for idx in range(len(FEAT))]
print("predictions : ",predicitons)
print("\n")


#TEST :


true_labels=[label for label in LABELS_STR]
print("Reality :",true_labels)

print("\nevals :", [evaluation(true_labels,predicitons,label) for label in LABELS_STR.unique()])

print("\nAccuracy : ",accuracy(true_labels,predicitons))
print("sklearn accuracy : ",sklearn.metrics.accuracy_score(true_labels,predicitons))

print("\nPrecision : ",precision(true_labels,predicitons))
print("sklearn precision : ",sklearn.metrics.precision_score(true_labels,predicitons,average="macro"))

print("\nRecall : ",recall(true_labels,predicitons))
print("sklearn recall : ",sklearn.metrics.recall_score(true_labels,predicitons,average="macro"))

print("\nF1 Score : ",f1_score(true_labels,predicitons))
print("sklearn F1 Score :",sklearn.metrics.f1_score(true_labels,predicitons,average="macro"))