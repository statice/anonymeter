# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Data pre-processing and transformations for the privacy evaluators."""
import logging
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def _encode_categorical(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode dataframes with categorical values keeping label consistend."""
    encoded = pd.concat((df1, df2), keys=["df1", "df2"])

    for col in encoded.columns:
        encoded[col] = LabelEncoder().fit_transform(encoded[col])

    return encoded.loc["df1"], encoded.loc["df2"]


def _scale_numerical(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale dataframes with *only* numerical values."""
    df1_min, df1_max = df1.min(), df1.max()
    df2_min, df2_max = df2.min(), df2.max()

    mins = df1_min.where(df1_min < df2_min, df2_min)
    maxs = df1_max.where(df1_max > df2_max, df2_max)
    ranges = maxs - mins

    if any(ranges == 0):
        cnames = ", ".join(ranges[ranges == 0].index.values)
        logger.debug(
            f"Numerical column(s) {cnames} have a null-range: all elements "
            "have the same value. These column(s) won't be scaled."
        )
        ranges[ranges == 0] = 1

    df1_scaled = df1.apply(lambda x: x / ranges[x.name])
    df2_scaled = df2.apply(lambda x: x / ranges[x.name])

    return df1_scaled, df2_scaled


def mixed_types_transform(
    df1: pd.DataFrame, df2: pd.DataFrame, num_cols: List[str], cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combination of an encoder and a scaler to treat mixed type data.

    Numerical columns are scaled by dividing them by their range across both
    datasets, so that the difference between any two values within a column will
    be smaller than or equal to one:
    x -> x' = x /  max{max(x), max(x_other)} - min{min(x), min(x_other)}

    Categorical columns are label encoded. This encoding is based on the
    `statice.preprocessing.encoders.DataframeEncoder` fitted on the firts
    dataframe, and applied to both of them.

    Parameters
    ----------
    df1: pd.DataFrame.
        Input DataFrame. This dataframe will be used to fit the DataframeLabelEncoder.
    df2: pd.DataFrame.
        Second input DataFrame.
    num_cols: list[str].
        Names of the numerical columns to be processed.
    cat_cols: list[str].
        Names of the  columns to be processed.

    Returns
    -------
    trans_df1: pd.DataFrame.
        Transformed df1.
    trans_df2: pd.DataFrame.
        Transformed df2.

    """
    if not set(df1.columns) == set(df2.columns):
        raise ValueError(f"Input dataframes have different columns. df1: {df1.columns}, df2: {df2.columns}.")

    if not set(num_cols + cat_cols) == set(df1.columns):
        raise ValueError(
            f"Dataframes columns {df1.columns} do not match "
            "with `num_cols` and `cat_cols`.\n"
            f"num_cols: {num_cols}\n"
            f"cat_cols: {cat_cols}"
        )

    df1_num, df2_num = pd.DataFrame(), pd.DataFrame()
    if len(num_cols) > 0:
        df1_num, df2_num = _scale_numerical(df1[num_cols], df2[num_cols])

    df1_cat, df2_cat = pd.DataFrame(), pd.DataFrame()
    if len(cat_cols) > 0:
        df1_cat, df2_cat = _encode_categorical(df1[cat_cols], df2[cat_cols])

    df1_out = pd.concat([df1_num, df1_cat], axis=1)[df1.columns]

    df2_out = pd.concat([df2_num, df2_cat], axis=1)[df2.columns]
    return df1_out, df2_out
