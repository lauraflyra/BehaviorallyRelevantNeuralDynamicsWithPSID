import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_funcs_py_neuromodulation import read_BIDS_data
import os
import PSID

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
ecog_right_0 = data[3, :].reshape(-1,1)
#take movement left clean
mov_left_clean = data[-1,:].reshape(-1,1)

# TODO: before giving data to PSID, separate into test and training set so we can test prediction with idSys.predict(yTest)


idSys = PSID.PSID(ecog_right_0, mov_left_clean, 4, 3, 5)

idSys