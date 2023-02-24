# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest

from anonymeter.neighbors.mixed_types_kneighbors import MixedTypeKNeighbors, gower_distance
from tests.fixtures import get_adult

rng = np.random.default_rng()


def test_mixed_type_kNN():
    df = get_adult("ori", n_samples=10)
    nn = MixedTypeKNeighbors().fit(df)
    shuffled_idx = rng.integers(10, size=10)
    dist, ids = nn.kneighbors(df.iloc[shuffled_idx], n_neighbors=1, return_distance=True)
    np.testing.assert_equal(ids.flatten(), shuffled_idx)
    np.testing.assert_equal(dist, 0)


def test_mixed_type_kNN_numerical():
    ori = pd.DataFrame([[0.0, "a"], [0.2, "a"], [0.15, "a"], [0.1, "a"]])
    syn = pd.DataFrame([[0.01, "a"]])
    nn = MixedTypeKNeighbors().fit(ori)
    ids = nn.kneighbors(syn, n_neighbors=4, return_distance=False)
    np.testing.assert_equal(ids, [[0, 3, 2, 1]])


def test_mixed_type_kNN_numerical_scaling():
    ori = pd.DataFrame([[0.0, "a"], [0.2, "a"], [0.15, "a"], [0.1, "a"]])

    # this is equal to the min value in the fitted dataframe.
    # The distance to the 2nd record in ori will be maximal.
    syn = pd.DataFrame([[0.0, "a"]])
    nn = MixedTypeKNeighbors().fit(ori)
    dist, ids = nn.kneighbors(syn, n_neighbors=4, return_distance=True)
    np.testing.assert_equal(ids, [[0, 3, 2, 1]])
    np.testing.assert_equal(dist[ids == 1], 1)


@pytest.mark.parametrize("n_neighbors, n_queries", [(1, 10), (3, 5)])
def test_mixed_type_kNN_shape(n_neighbors, n_queries):
    df = get_adult("ori", n_samples=10)
    nn = MixedTypeKNeighbors(n_neighbors=n_neighbors).fit(df)
    ids = nn.kneighbors(df.head(n_queries))
    assert ids.shape == (n_queries, n_neighbors)

    nn = MixedTypeKNeighbors().fit(df)
    ids = nn.kneighbors(df.head(n_queries), n_neighbors=n_neighbors)
    assert ids.shape == (n_queries, n_neighbors)


@pytest.mark.parametrize(
    "r0, r1, expected",
    [
        ([0, 1, 0, 0], [0, 1, 0, 0], 0),
        ([1, 1, 0, 0], [0, 1, 0, 0], 1),
        ([1, 1, 1, 0], [0, 1, 0, 0], 2),
        ([1, 0, 1, 0], [1, 1, 0, 1], 3),
        ([1, 0, 1, 0], [0, 1, 0, 1], 4),
    ],
)
def test_gower_distance(r0, r1, expected):
    r0, r1 = np.array(r0), np.array(r1)
    dist = gower_distance(r0=r0, r1=r1, cat_cols_index=0)
    np.testing.assert_equal(dist, expected)

    # numerical and categorical should behave the same
    dist = gower_distance(r0=r0, r1=r1, cat_cols_index=4)
    np.testing.assert_equal(dist, expected)


def test_gower_distance_numerical():
    r0, r1 = rng.random(size=10), rng.random(size=10)
    dist = gower_distance(r0=r0, r1=r1, cat_cols_index=10)
    np.testing.assert_almost_equal(dist, np.sum(np.abs(r0 - r1)))
