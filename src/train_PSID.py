import PSID
from PSID.evaluation import evalPrediction
import numpy as np
import matplotlib.pyplot as plt

def train(features, behavior, nx, n1, i, eval_metrics = "R2", train_whole_data = False):

    if not train_whole_data:
        train_idx_features = np.arange(np.round(0.5 * features.shape[0]), dtype=int)
        test_idx_features = np.arange(1 + train_idx_features[-1], features.shape[0])

        feat_train = features[train_idx_features]
        feat_test =features[test_idx_features]
        mov_features_train = behavior[train_idx_features]
        mov_features_test = behavior[test_idx_features]

        idSys = PSID.PSID(feat_train, mov_features_train, nx,  n1, i)

        mov_features_test_pred, feat_test_pred, x_feat_test_pred = idSys.predict(feat_test)

        eval_result = evalPrediction(mov_features_test, mov_features_test_pred, eval_metrics)

        print('Behavior decoding r2:\n  PSID => {:.3g}'.format(np.mean(eval_result)))

        return idSys, mov_features_test_pred, mov_features_test, test_idx_features, eval_result
    else:
        idSys = PSID.PSID(features, behavior, nx, n1, i)

        return idSys



def plot_movement_pred(df, mov_features_test_pred, mov_features_test, test_idx_features, eval_result):

    time_array = df['time'].to_numpy().reshape(-1, 1)[test_idx_features]
    plt.plot(time_array, mov_features_test.reshape(-1,), label = 'data')
    plt.plot(time_array, mov_features_test_pred.reshape(-1,), label = 'prediction')
    plt.title("Hand decoding for test data with bandpass features \nfrom all ECOG and STN channels as input, R^2 = {:.3g}".format(np.mean(eval_result)), fontsize = 15)
    plt.xlabel("Time [ms]", fontsize = 15)
    plt.ylabel("Grip force [a.u]", fontsize = 15)
    plt.legend(fontsize = 12)
    plt.show()

    return