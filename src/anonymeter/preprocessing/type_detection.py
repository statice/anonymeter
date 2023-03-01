# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
from typing import Dict, List

import pandas as pd


def detect_col_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify numerical and non-numerical columns in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    Dict[str: List[str]]
        Dictionary with column names separated by types. Key of the dictionary are
        'num' or 'cat' (numerical and non-numerical, that is categorical, resp.).
        Values are lists of column names.

    """
    num_cols: List[str] = list(df.select_dtypes("number").columns.values)
    cat_cols: List[str] = [cn for cn in df.columns.values if cn not in num_cols]

    return {"num": sorted(num_cols), "cat": sorted(cat_cols)}


def detect_consistent_col_types(df1: pd.DataFrame, df2: pd.DataFrame):
    """Detect colum types for a pair dataframe an check that they are the same.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Input dataframe
    df2 : pandas.DataFrame
        Input dataframe

    Returns
    -------
    Dict[str: List[str]]
        Dictionary with column names separated by types. Key of the dictionary are
        'num' or 'cat' (numerical and non-numerical, that is categorical, resp.).
        Values are lists of column names.

    """
    ctypes1 = detect_col_types(df1)

    if ctypes1 != detect_col_types(df2):
        raise RuntimeError("Input dataframes have different column names/types.")

    return ctypes1
