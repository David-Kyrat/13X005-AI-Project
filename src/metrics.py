
#TODO : COMMENT
#TODO : Adapt to the logistic regression implementation (columns names...)

def evaluation(test_data,predict_labels):
    """
    This function evaluates a prediction by calculating the true/false positive/negative, which are necessary for the below metrics.

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: list
        predicted labels
    
    Return value
    ------------
    tuple of int (4 elements wich are the true/false positive/negative values)
    """

    last_column_name = test_data.columns[-1]
    test_labels = test_data[last_column_name]

    true_negative=sum(real == 0 and pred== 0 for real, pred in zip(test_labels, predict_labels))
    true_positive=sum(real == 1 and pred== 1 for real, pred in zip(test_labels, predict_labels))
    false_negative=sum(real == 1 and pred== 0 for real, pred in zip(test_labels, predict_labels))
    false_positive=sum(true == 0 and pred== 1 for true, pred in zip(test_labels, predict_labels))

    return true_positive, false_positive, true_negative, false_negative


def accuracy(true_labels, predicted_labels):
    """
    This function implements the accuracy metric wich measure the proportion of correct predictions considering all predictions

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: list
        predicted labels
    
    Return value
    ------------
    float number wich is the accuracy proportion value between 0 and 1.
    """

    correct_predictions = sum(true == pred for true, pred in zip(true_labels, predicted_labels))
    total_instances = len(true_labels)
    return correct_predictions / total_instances


def precision(test_data,predict_labels):
    """
    This function implements the precision metric wich measure the proportion of false positive predictions considering every positive predictions

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: list
        predicted labels
    
    Return value
    ------------
    float number wich is the precision value between 0 and 1.
    """

    true_positive, false_positive, _, _=evaluation(test_data,predict_labels)
    return true_positive / (true_positive + false_positive)


def recall(test_data,predict_labels):
    """
    This function implements the precision metric wich measure the proportion of uncorrect negatives predictions (recall of 1 means the number of fale negatives is null)

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: list
        predicted labels
    
    Return value
    ------------
    float number wich is the recall value between 0 and 1.
    """

    true_positive, _, _, false_negative=evaluation(test_data,predict_labels)
    return true_positive / (true_positive + false_negative)


def f1_score(test_data,predict_labels):
    """
    This function implements the f1 score metric wich is a balanced measure between precision and recall that measures errors more broadly

    Parameters
    ----------
    `test_data`: pandas.DataFrame
        test dataset
    `predict_labels`: list
        predicted labels
    
    Return value
    ------------
    float number wich is the f1 score value between 0 and 1.
    """

    prec=precision(test_data,predict_labels)
    rec=recall(test_data,predict_labels)
    return 2 * (prec * rec) / (prec + rec)



# ================================================================
# ======================= TEST:==================================
# ================================================================

#TODO : utiliser le dataset test pour ensuite appliquer les métriques
