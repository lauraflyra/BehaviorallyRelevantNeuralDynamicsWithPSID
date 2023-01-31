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