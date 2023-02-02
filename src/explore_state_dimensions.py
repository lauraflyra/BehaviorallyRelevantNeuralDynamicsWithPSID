import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PSID
from PSID.evaluation import evalPrediction

# explore dimensions of behaviorally relevant dynamics
# in here all latent space dimensions will be used to prioritize behavior, meaning n1 = nx

PATH_FEATURES = "/home/lauraflyra/Documents/BCCN/Lab_Rotation_USC/Code/Data/py_neuromodulation_derivatives/sub-000_ses-right_task-force_run-3/sub-000_ses-right_task-force_run-3_FEATURES.csv"
data_features = pd.read_csv(PATH_FEATURES, index_col=0)

def cross_validation_split(data, k_folds, k):
    size_fold = int(data.shape[0]/k_folds)
    test_set = data[k*size_fold:k*size_fold+size_fold]
    training_set = np.delete(data, np.arange(k*size_fold,k*size_fold+size_fold),axis=0)
    return test_set, training_set


def cross_validation(neural_data, behavior, k_folds, nx, n1, i, metrics = "CC"):
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


feature_df = pd.concat([data_features.filter(like='STN'),data_features.filter(like='ECOG')], axis = 1).filter(like='bandpass_activity').to_numpy()
behavior_df = data_features["MOV_LEFT_CLEAN"].to_numpy().reshape(-1, 1)

K_FOLDS = 5
N_DIMS = 10

eval_over_dims = np.zeros((N_DIMS, K_FOLDS))
behavior_dims_latent = np.linspace(2, 20, N_DIMS, dtype=int)
for dim in range(N_DIMS):
    nx = n1 = behavior_dims_latent[dim]
    i = max(behavior_dims_latent)
    eval_over_dims[dim,:] = cross_validation(feature_df, behavior_df, K_FOLDS, nx, n1, i)
















