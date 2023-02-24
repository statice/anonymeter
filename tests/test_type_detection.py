# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest

from anonymeter.preprocessing.type_detection import detect_col_types, detect_consistent_col_types

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "df, expected",
    [
        (pd.DataFrame({"num": rng.random(5), "cat": list("abcde")}), {"cat": ["cat"], "num": ["num"]}),
        (pd.DataFrame({"num1": rng.random(5), "num2": [1, 2, 3, 4, 5]}), {"cat": [], "num": ["num1", "num2"]}),
        (
            pd.DataFrame({"num1": rng.random(5), "num2": [1, 2, 3, 4, 5]}).astype("object"),
            {"cat": ["num1", "num2"], "num": []},
        ),
        (
            pd.DataFrame({"cat1": list("abcde"), "cat2": ["1", "2", "3", "4", "5"]}),
            {"cat": ["cat1", "cat2"], "num": []},
        ),
    ],
)
def test_detect_col_types(df, expected):
    ctypes = detect_col_types(df=df)
    assert ctypes == expected


def test_detect_col_types_consistent():
    df1 = pd.DataFrame({"num": rng.random(5), "cat": list("abcde")})
    df2 = pd.DataFrame({"num": rng.random(5), "cat": list("fghil")})
    assert detect_consistent_col_types(df1, df2) == {"cat": ["cat"], "num": ["num"]}


def test_detect_col_types_consistent_raises():
    df1 = pd.DataFrame({"num": rng.random(5), "cat": list("abcde")})
    df2 = pd.DataFrame({"num": [str(_) for _ in rng.random(5)], "cat": list("fghil")})
    with pytest.raises(RuntimeError):
        detect_consistent_col_types(df1, df2)
