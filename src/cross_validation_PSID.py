import numpy as np
import PSID
from PSID.evaluation import evalPrediction

def cross_validation_split(data, k_folds, k):
    size_fold = int(data.shape[0]/k_folds)
    test_set = data[k*size_fold:k*size_fold+size_fold]
    training_set = np.delete(data, np.arange(k*size_fold,k*size_fold+size_fold),axis=0)
    return test_set, training_set


def cross_validation(neural_data, behavior, k_folds, nx, n1, i, metrics = "R2"):
    """

    :param neural_data: time x neural data dimensions
    :param behavior: time x behavioral data dimensions
    :param k_folds: k fold CV
    :param nx: total dimension of the latent state
    :param n1: how many of those dimensions will be prioritizing behavior
    :param i: horizon, determines maximum n1 and nx: n1 <= nz * i. nx <= ny * i
    :param metrics: evaluation prediction metrics accepted by PSID
    :return:
    """
    eval_over_folds = np.zeros(k_folds)
    for k in range(k_folds):
        neural_data_test, neural_data_train = cross_validation_split(neural_data, k_folds, k)
        behavior_test, behavior_train = cross_validation_split(behavior, k_folds, k)

        idSys = PSID.PSID(neural_data_train, behavior_train, nx, n1, i)

        behavior_test_pred, neural_data_test_pred, x_test_pred = idSys.predict(neural_data_test)
        eval = evalPrediction(behavior_test, behavior_test_pred, metrics)

        print('Behavior decoding evaluation:\n  PSID => {:.3g}'.format(np.mean(eval)))
        eval_over_folds[k] = np.mean(eval)

    return eval_over_folds














