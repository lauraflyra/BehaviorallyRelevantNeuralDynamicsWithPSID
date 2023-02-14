"""
File with helper functions taken from https://github.com/neuromodulation/py_neuromodulation
Author: Timon Merk
"""


import os

import mne
import mne_bids
import numpy as np


_PathLike = str | os.PathLike


def read_BIDS_data(
        PATH_RUN: _PathLike | mne_bids.BIDSPath,
        BIDS_PATH: _PathLike | None = None,
        datatype: str = "ieeg",
        line_noise: int = 50,
) -> tuple[mne.io.Raw, np.ndarray, int | float, int, list | None, list | None]:
    """Given a run path and bids data path, read the respective data

    Parameters
    ----------
    PATH_RUN : string
    BIDS_PATH : string
    datatype : string

    Returns
    -------
    raw_arr : mne.io.RawArray
    raw_arr_data : np.ndarray
    fs : int
    line_noise : int
    """
    if isinstance(PATH_RUN, mne_bids.BIDSPath):
        bids_path = PATH_RUN
    else:
        bids_path = mne_bids.get_bids_path_from_fname(PATH_RUN)

    raw_arr = mne_bids.read_raw_bids(bids_path)
    coord_list, coord_names = get_coord_list(raw_arr)
    if raw_arr.info["line_freq"] is not None:
        line_noise = int(raw_arr.info["line_freq"])
    else:
        print(
            "Line noise is not available in the data, using value of {} Hz.".format(
                line_noise
            )
        )
    return (
        raw_arr,
        raw_arr.get_data(),
        raw_arr.info["sfreq"],
        line_noise,
        coord_list,
        coord_names,
    )

def get_coord_list(
    raw: mne.io.BaseRaw,
) -> tuple[list, list] | tuple[None, None]:
    montage = raw.get_montage()
    if montage is not None:
        coord_list = np.array(
            list(dict(montage.get_positions()["ch_pos"]).values())
        ).tolist()
        coord_names = np.array(
            list(dict(montage.get_positions()["ch_pos"]).keys())
        ).tolist()
    else:
        coord_list = None
        coord_names = None

    return coord_list, coord_names

def get_epochs(data, y_, epoch_len, sfreq, threshold=0):
    """Return epoched data.

    Parameters
    ----------
    data : np.ndarray
        array of extracted features of shape (n_samples, n_channels, n_features)
    y_ : np.ndarray
        array of labels e.g. ones for movement and zeros for
        no movement or baseline corr. rotameter data
    sfreq : int/float
        sampling frequency of data
    epoch_len : int
        length of epoch in seconds
    threshold : int/float
        (Optional) threshold to be used for identifying events
        (default=0 for y_tr with only ones
        and zeros)

    Returns
    -------
    epoch_ np.ndarray
        array of epoched ieeg data with shape (epochs,samples,channels,features)
    y_arr np.ndarray
        array of epoched event label data with shape (epochs,samples)
    """

    epoch_lim = int(epoch_len * sfreq)

    ind_mov = np.where(np.diff(np.array(y_ > threshold) * 1) == 1)[0]

    low_limit = ind_mov > epoch_lim / 2
    up_limit = ind_mov < y_.shape[0] - epoch_lim / 2

    ind_mov = ind_mov[low_limit & up_limit]

    epoch_ = np.zeros(
        [ind_mov.shape[0], epoch_lim, data.shape[1], data.shape[2]]
    )

    y_arr = np.zeros([ind_mov.shape[0], int(epoch_lim)])

    for idx, i in enumerate(ind_mov):

        epoch_[idx, :, :, :] = data[
            i - epoch_lim // 2 : i + epoch_lim // 2, :, :
        ]

        y_arr[idx, :] = y_[i - epoch_lim // 2 : i + epoch_lim // 2]

    return epoch_, y_arr
