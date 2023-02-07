import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PSID
from PSID.evaluation import evalPrediction
from src.cross_validation_PSID import cross_validation, cross_validation_split


def explore_latent_behavior_dims(neural, behavior, k_folds, n_dims, max_dim, i, cv_eval='R2'):
    """

    :param neural:
    :param behavior:
    :param k_folds:
    :param n_dims:
    :param max_dim:
    :param i:
    :param cv_eval:
    :return:
    """
    eval_over_dims = np.zeros((n_dims, k_folds))
    behavior_dims_latent = np.linspace(2, max_dim, n_dims, dtype=int)
    for dim in range(n_dims):
        nx = n1 = behavior_dims_latent[dim]
        eval_over_dims[dim, :] = cross_validation(neural, behavior, k_folds, nx, n1, i, metrics=cv_eval)

    return eval_over_dims, behavior_dims_latent


def plot_explorations(evals, behavior_dims_latent, i_s, feature_type='bandpass activity', channel_type='STN and ECOG',
                      metrics='R2', xgboost_mean=0.6, xgboost_std=0.05, plot_xgboost=True):
    for eval, i in zip(evals, i_s):
        plt.plot(behavior_dims_latent, np.mean(eval, axis=1), label="i = {}".format(i))
        plt.fill_between(behavior_dims_latent,
                         np.mean(eval, axis=1) - np.std(eval, axis=1),
                         np.mean(eval, axis=1) + np.std(eval, axis=1),
                         alpha=0.5)
    if plot_xgboost:
        plt.axhline(y=xgboost_mean, color='black', linestyle='-', label='XGBOOST')
        plt.axhline(y=xgboost_mean + xgboost_std, color='black', linestyle='-.',
                    label='XGBOOST + std', alpha=0.6)
        plt.axhline(y=xgboost_mean - xgboost_std, color='black', linestyle='-.',
                    label='XGBOOST - std', alpha=0.6)

    # plt.ylim(0, 1)
    plt.ylabel("Decoding " + metrics, fontsize=12)
    plt.xlabel("State dimension", fontsize=12)
    plt.legend()
    plt.title(
        "Decoding " + metrics + " of behavior from neural activity \nfor " + feature_type + " from " + channel_type,
        fontsize=12)
    plt.show()
