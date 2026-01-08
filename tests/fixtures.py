# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details..


import os
from typing import Optional

import pandas as pd

TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_adult(which: str, deduplicate_on: Optional[list[str]] = None, n_samples: Optional[int] = None) -> pd.DataFrame:
    """Fixture for the adult dataset.

    For details see:
    https://archive.ics.uci.edu/ml/datasets/adult

    Parameters
    ----------
    which : str, in ['ori', 'syn']
        Whether to return the "original" or "synthetic" samples.
    deduplicate_on: list of str
        A list of columns based on which we'd drop duplicates in the samples.
    n_samples : int
        Number of sample records to return.
        If `None` - return all samples.

    Returns
    -------
    df : pd.DataFrame
        Adult dataframe.
    """
    if which == "ori":
        fname = "adults_ori.csv"
    elif which == "syn":
        fname = "adults_syn.csv"
    else:
        raise ValueError(f"Invalid value {which} for parameter `which`. Available are: 'ori' or 'syn'.")

    samples = pd.read_csv(os.path.join(TEST_DIR_PATH, "datasets", fname))
    if deduplicate_on:
        samples = samples.drop_duplicates(subset=deduplicate_on)
    return samples.iloc[:n_samples]
