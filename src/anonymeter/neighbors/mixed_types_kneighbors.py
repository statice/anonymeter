# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Nearest neighbor search for mixed type data."""
import logging
from math import fabs, isnan
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import float64, int64, jit

from anonymeter.preprocessing.transformations import mixed_types_transform
from anonymeter.preprocessing.type_detection import detect_consistent_col_types

logger = logging.getLogger(__name__)


@jit(nopython=True, nogil=True)
def gower_distance(r0: np.ndarray, r1: np.ndarray, cat_cols_index: np.ndarray) -> float64:
    r"""Distance between two records inspired by the Gower distance [1].

    To handle mixed type data, the distance is specialized for numerical (continuous)
    and categorical data. For numerical records, we use the L1 norm,
    computed after the columns have been normalized so that :math:`d(a_i, b_i)\leq 1`
    for every :math:`a_i`, :math:`b_i`. For categorical, :math:`d(a_i, b_i)` is 1,
    if the entries :math:`a_i`, :math:`b_i` differ, else, it is 0.

    Notes
    -----
    To keep the balance between numerical and categorical values, the input records
    have to be properly normalized. Their numerical part need to be scaled so that
    the difference between any two values of a column (from both dataset) is *at most* 1.

    References
    ----------
    [1]. `Gower (1971) "A general coefficient of similarity and some of its properties.
    <https://www.jstor.org/stable/2528823?seq=1>`_

    Parameters
    ----------
    r0 : np.array
        Input array of shape (D,).
    r1 : np.array
        Input array of shape (D,).
    cat_cols_index : int
        Index delimiting the categorical columns in r0/r1 if present. For example,
        ``r0[:cat_cols_index]`` are the numerical columns, and ``r0[cat_cols_index:]`` are
        the categorical ones. For a fully numerical dataset, use ``cat_cols_index =
        len(r0)``. For a fully categorical one, set ``cat_cols_index`` to 0.

    Returns
    -------
    float
        distance between the records.

    """
    dist = 0.0

    for i in range(len(r0)):

        if isnan(r0[i]) and isnan(r1[i]):
            dist += 1

        else:
            if i < cat_cols_index:
                dist += fabs(r0[i] - r1[i])

            else:
                if r0[i] != r1[i]:
                    dist += 1
    return dist


@jit(nopython=True, nogil=True)
def _nearest_neighbors(queries, candidates, cat_cols_index, n_neighbors):
    r"""For every element of ``queries``, find its nearest neighbors in ``candidates``.

    Parameters
    ----------
    queries : np.ndarray
        Input array of shape (Nx, D).
    candidates : np.ndarray
        Input array of shape (Ny, D).
    n_neighbors : int
        Determines the number of closest neighbors per entry to be returned.
    cat_cols_idx : int
        Index delimiting the categorical columns in X/Y, if present.

    Returns
    -------
    idx : np.ndarray[int]
        Array of shape (Nx, n_neighbors). For each element in ``queries``,
        this array contains the indices of the closest neighbors in
        ``candidates``. That is, ``candidates[idx[i]]`` are the elements of
        ``candidates`` that are closer to ``queries[i]``.
    lps : np.ndarray[float]
        Array of shape (Nx, n_neighbors). This array containing the distances
        between the record pairs identified by idx.

    """
    idx = np.zeros((queries.shape[0], n_neighbors), dtype=int64)
    dists = np.zeros((queries.shape[0], n_neighbors), dtype=float64)

    for ix in range(queries.shape[0]):

        dist_ix = np.zeros((candidates.shape[0]), dtype=float64)

        for iy in range(candidates.shape[0]):

            dist_ix[iy] = gower_distance(r0=queries[ix], r1=candidates[iy], cat_cols_index=cat_cols_index)

        close_match_idx = dist_ix.argsort()[:n_neighbors]
        idx[ix] = close_match_idx
        dists[ix] = dist_ix[close_match_idx]

    return idx, dists


class MixedTypeKNeighbors:
    """Nearest neighbor algorithm for mixed type data.

    To handle mixed type data, we use a distance function inspired by the Gower similarity.
    The distance is specialized for numerical (continuous) and categorical data. For
    numerical records, we use the L1 norm, computed after the columns have been
    normalized so that :math:`d(a_i, b_i) <= 1` for every :math:`a_i`, :math:`b_i`.
    For categorical, :math:`d(a_i, b_i)` is 1, if the entries :math:`a_i`, :math:`b_i`
    differ, else, it is 0.

    References
    ----------
    [1]. `Gower (1971) "A general coefficient of similarity and some of its properties.
    <https://www.jstor.org/stable/2528823?seq=1>`_

    Parameters
    ----------
    n_neighbors : int, default is 5
        Determines the number of closest neighbors per entry to be returned.
    n_jobs : int, default is -2
        Number of jobs to use. It follows joblib convention, so that ``n_jobs = -1``
        means all available cores.

    """

    def __init__(self, n_neighbors: int = 5, n_jobs: int = -2):
        self._n_neighbors = n_neighbors
        self._n_jobs = n_jobs

    def fit(self, candidates: pd.DataFrame, ctypes: Optional[Dict[str, List[str]]] = None):
        """Prepare for nearest neighbor search.

        Parameters
        ----------
        candidates : pd.DataFrame
            Dataset containing the records one would find the neighbors in.
        ctypes : dict, optional.
            Dictionary specifying which columns in X should be treated as
            continuous and which should be treated as categorical. For example,
            ``ctypes = {'num': ['distance'], 'cat': ['color']}`` specify the types
            of a two column dataset.

        """
        self._candidates = candidates
        self._ctypes = ctypes
        return self

    def kneighbors(
        self, queries: pd.DataFrame, n_neighbors: Optional[int] = None, return_distance: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Find the nearest neighbors for a set of query points.

        Note
        ----
        The search is performed in a brute-force fashion. For large datasets
        or large number of query points, the search for nearest neighbor will
        become very slow.

        Parameters
        ----------
        queries : pd.DataFrame
            Query points for the nearest neighbor searches.
        n_neighbors : int, default is None
            Number of neighbors required for each sample.
            The default is the value passed to the constructor.
        return_distance : bool, default is False
            Whether or not to return the distances of the neigbors or
            just the indexes.

        Returns
        -------
        np.narray of shape (df.shape[0], n_neighbors)
            Array with the indexes of the elements of the fit dataset closer to
            each element in the query dataset.
        np.narray of shape (df.shape[0], n_neighbors)
            Array with the distances of the neighbors pairs. This is optional and
            it is returned only if ``return_distances`` is ``True``

        """
        if n_neighbors is None:
            n_neighbors = self._n_neighbors

        if n_neighbors > self._candidates.shape[0]:
            logger.warning(
                f"Parameter ``n_neighbors``={n_neighbors} cannot be "
                f"larger than the size of the training data {self._candidates.shape[0]}."
            )
            n_neighbors = self._candidates.shape[0]

        if self._ctypes is None:
            self._ctypes = detect_consistent_col_types(df1=self._candidates, df2=queries)
        candidates, queries = mixed_types_transform(
            df1=self._candidates, df2=queries, num_cols=self._ctypes["num"], cat_cols=self._ctypes["cat"]
        )

        cols = self._ctypes["num"] + self._ctypes["cat"]
        queries = queries[cols].values
        candidates = candidates[cols].values

        with Parallel(n_jobs=self._n_jobs, backend="threading") as executor:

            res = executor(
                delayed(_nearest_neighbors)(
                    queries=queries[ii : ii + 1],
                    candidates=candidates,
                    cat_cols_index=len(self._ctypes["num"]),
                    n_neighbors=n_neighbors,
                )
                for ii in range(queries.shape[0])
            )

            indexes, distances = zip(*res)
            indexes, distances = np.vstack(indexes), np.vstack(distances)

        if return_distance:
            return distances, indexes

        return indexes
