import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_funcs_py_neuromodulation import read_BIDS_data
import os
import PSID
from PSID.evaluation import evalPrediction

# --------------------------------------------------------------- #
# TRY WITH RAW_DATA #

SCRIPT_DIR = os.path.dirname(os.path.abspath(''))
SCRIPT_DIR = os.path.join(SCRIPT_DIR, "Code")

sub = "000"
ses = "right"
task = "force"
run = 3
datatype = "ieeg"

# Define run name and access paths in the BIDS format.
RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

PATH_RUN = os.path.join(
    (os.path.join(SCRIPT_DIR, "Data", "raw_data")),
    f"sub-{sub}",
    f"ses-{ses}",
    datatype,
    RUN_NAME,
)
PATH_BIDS = os.path.join(SCRIPT_DIR, "Data", "raw_data")

(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = read_BIDS_data(
    PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
)

# get the list of channel names
print(raw.ch_names)

# data has shape (13, 282000)
# take first ECOG channel:
ecog_right_0 = data[3, :].reshape(-1, 1)
# take movement left clean
mov_left_clean = data[-1, :].reshape(-1, 1)

train_idx = np.arange(np.round(0.5 * data.shape[1]), dtype=int)
test_idx = np.arange(1 + train_idx[-1], data.shape[1])

ecog_train = ecog_right_0[train_idx, :]
ecog_test = ecog_right_0[test_idx, :]

mov_train = mov_left_clean[train_idx, :]
mov_test = mov_left_clean[test_idx, :]

idSys = PSID.PSID(ecog_train, mov_train, 20, 20, 30)

# Predict behavior using the learned model
mov_test_pred, ecog_test_pred, x_test_pred = idSys.predict(ecog_test)

R2 = evalPrediction(mov_test, mov_test_pred, "CC")

print('Behavior decoding R2:\n  PSID => {:.3g}'.format(np.mean(R2)))

plt.plot(mov_test.reshape(-1,), label = 'data')
plt.plot(mov_test_pred.reshape(-1,), label = 'prediction')
plt.legend()
plt.show()

# ----------------------------------------------------- #
# TRY WITH FEATURES #

PATH_FEATURES = "/home/lauraflyra/Documents/BCCN/Lab_Rotation_USC/Code/Data/py_neuromodulation_derivatives/sub-000_ses-right_task-force_run-3/sub-000_ses-right_task-force_run-3_FEATURES.csv"

data_features = pd.read_csv(PATH_FEATURES, index_col=0)

train_idx_features = np.arange(np.round(0.5 * data_features.shape[0]), dtype=int)
test_idx_features = np.arange(1 + train_idx_features[-1], data_features.shape[0])

# stn_fft_feat_train = data_features.filter(like='STN').filter(like="bandpass_activity").to_numpy()[train_idx_features]
# stn_fft_feat_test = data_features.filter(like='STN').filter(like="bandpass_activity").to_numpy()[test_idx_features]

stn_fft_feat_train = pd.concat([data_features.filter(like='STN'),data_features.filter(like='ECOG')], axis = 1).filter(like='bandpass_activity').to_numpy()[train_idx_features]
stn_fft_feat_test = pd.concat([data_features.filter(like='STN'),data_features.filter(like='ECOG')], axis = 1).filter(like='bandpass_activity').to_numpy()[test_idx_features]

# TODO: try to drop bandpass in low, high gamma and HFA: https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe
# based on the correlation plot from example_BIDS, those three ranges are not so correlated to the other ones.
# TODO: run correlation plot of features and movement trances for example_BIDS data, maybe i'll see something?

mov_features_train = data_features["MOV_LEFT_CLEAN"].to_numpy().reshape(-1, 1)[train_idx_features]
mov_features_test = data_features["MOV_LEFT_CLEAN"].to_numpy().reshape(-1, 1)[test_idx_features]

idSys = PSID.PSID(stn_fft_feat_train, mov_features_train, 30, 10, 50)

# TODO: explore other dimensions??
# TODO: appropriate renaming
# TODO: choose other combinations of features from py_neuromodulation

# Predict behavior using the learned model
mov_features_test_pred, stn_fft_feat_test_pred, x_feat_test_pred = idSys.predict(stn_fft_feat_test)

R2_feat = evalPrediction(mov_features_test, mov_features_test_pred, "CC")

print('Behavior decoding R2:\n  PSID => {:.3g}'.format(np.mean(R2_feat)))

plt.plot(mov_features_test.reshape(-1,), label = 'data')
plt.plot(mov_features_test_pred.reshape(-1,), label = 'prediction')
plt.legend()
plt.show()
