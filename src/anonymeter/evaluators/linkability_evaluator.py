# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the linkability risk."""
import logging
from typing import Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd

from anonymeter.neighbors.mixed_types_kneighbors import MixedTypeKNeighbors
from anonymeter.stats.confidence import EvaluationResults, PrivacyRisk

logger = logging.getLogger(__name__)


class LinkabilityIndexes:
    """Utility class to store indexes from linkability attack.

    Parameters
    ----------
    idx_0 : np.ndarray
        Array containing the result of the nearest neighbor search
        between the first original dataset and the synthetic data.
        Rows correspond to original records and the i-th column
        contains the index of the i-th closest synthetic record.
    idx_1 : np.ndarray
        Array containing the result of the nearest neighbor search
        between the second original dataset and the synthetic data.
        Rows correspond to original records and the i-th column
        contains the index of the i-th closest synthetic record.

    """

    def __init__(self, idx_0: np.ndarray, idx_1: np.ndarray):
        self._idx_0 = idx_0
        self._idx_1 = idx_1

    def find_links(self, n_neighbors: int) -> Dict[int, Set[int]]:
        """Return synthetic records that link originals in the split datasets.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors considered for the link search.

        Returns
        -------
        Dict[int, Set[int]]
            Dictionary mapping the index of the linking synthetic record
            to the index of the linked original record.

        """
        if n_neighbors > self._idx_0.shape[0]:
            logger.warning(f"Neighbors too large ({n_neighbors}, using {self._idx_0.shape[0]}) instead.")
            n_neighbors = self._idx_0.shape[0]

        if n_neighbors < 1:
            raise ValueError(f"Invalid neighbors value ({n_neighbors}): must be positive.")

        links = {}
        for ii, (row0, row1) in enumerate(zip(self._idx_0, self._idx_1)):
            joined = set(row0[:n_neighbors]) & set(row1[:n_neighbors])
            if len(joined) > 0:
                links[ii] = joined

        return links

    def count_links(self, n_neighbors: int) -> int:
        """Count successfully linked records.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors considered for the link search.

        Returns
        -------
        int
            Number of target records for which the synthetic dataset
            has provided the attacker wth means to link them.

        """
        links = self.find_links(n_neighbors=n_neighbors)
        return _count_links(links)


def _count_links(links: Dict[int, Set[int]]) -> int:
    """Count links."""
    linkable: Set[int] = set()

    for ori_idx in links.keys():
        linkable = linkable | {ori_idx}

    return len(linkable)


def _random_links(n_synthetic: int, n_attacks: int, n_neighbors: int) -> np.ndarray:
    rng = np.random.default_rng()

    return np.array([rng.choice(n_synthetic, size=n_neighbors, replace=False) for _ in range(n_attacks)])


def _random_linkability_attack(n_synthetic: int, n_attacks: int, n_neighbors: int) -> LinkabilityIndexes:
    idx_0 = _random_links(n_synthetic=n_synthetic, n_attacks=n_attacks, n_neighbors=n_neighbors)
    idx_1 = _random_links(n_synthetic=n_synthetic, n_attacks=n_attacks, n_neighbors=n_neighbors)

    return LinkabilityIndexes(idx_0=idx_0, idx_1=idx_1)


def _find_nn(syn: pd.DataFrame, ori: pd.DataFrame, n_jobs: int, n_neighbors: int) -> np.ndarray:
    nn = MixedTypeKNeighbors(n_jobs=n_jobs, n_neighbors=n_neighbors)

    if syn.ndim == 1:
        syn = syn.to_frame()

    if ori.ndim == 1:
        ori = ori.to_frame()

    nn.fit(syn)

    return cast(np.ndarray, nn.kneighbors(ori, return_distance=False))


def _linkability_attack(
    ori: pd.DataFrame,
    syn: pd.DataFrame,
    n_attacks: int,
    aux_cols: Tuple[List[str], List[str]],
    n_neighbors: int,
    n_jobs: int,
) -> LinkabilityIndexes:
    targets = ori.sample(n_attacks, replace=False)

    idx_0 = _find_nn(syn=syn[aux_cols[0]], ori=targets[aux_cols[0]], n_neighbors=n_neighbors, n_jobs=n_jobs)
    idx_1 = _find_nn(syn=syn[aux_cols[1]], ori=targets[aux_cols[1]], n_neighbors=n_neighbors, n_jobs=n_jobs)

    return LinkabilityIndexes(idx_0=idx_0, idx_1=idx_1)


class LinkabilityEvaluator:
    r"""Measure the linkability risk created by a synthetic dataset.

    The linkability risk is measured from the success of a linkability attack.
    The attack is modeled along the following scenario. The attacker posesses
    two datasets, both of which share some columns with the *original* dataset
    that was used to generate the synthetic data. Those columns will be
    referred to as *auxiliary columns*. The attacker's aim is then to use the
    information contained in the synthetic data to connect these two datasets,
    i.e. to find records that belong to the same individual.

    To model this attack, the original dataset is split vertically into two
    parts. Then we try to reconnect the two parts using the synthetic data
    by looking for the closest neighbors of the split original records in
    the synthetic data. If both splits of an original record have the same
    closest synthetic neighbor, they are linked together. The more original
    records get relinked in this manner the more successful the attack.


    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe containing original data.
    syn : pd.DataFrame
        Dataframe containing synthetic data. It has to have
        the same columns as df_ori.
    aux_cols : tuple of two lists of strings or tuple of int, optional
        Features of the records that are given to the attacker as auxiliary
        information.
    n_attacks : int, default is 500.
        Number of records to attack. If None each record in the original
        dataset will be attacked.
    n_neighbors : int, default is 1
        The number of closest neighbors to include in the analysis. The
        default of 1 means that the linkability attack is considered
        successful only if the two original record split have the same
        synthetic record as closest neighbor.
    control : pd.DataFrame (optional)
        Independent sample of original records **not** used to create the
        synthetic dataset. This is used to evaluate the excess privacy risk.
    """

    def __init__(
        self,
        ori: pd.DataFrame,
        syn: pd.DataFrame,
        aux_cols: Tuple[List[str], List[str]],
        n_attacks: Optional[int] = 500,
        n_neighbors: int = 1,
        control: Optional[pd.DataFrame] = None,
    ):
        self._ori = ori
        self._syn = syn
        self._n_attacks = n_attacks if n_attacks is not None else ori.shape[0]
        self._aux_cols = aux_cols
        self._n_neighbors = n_neighbors
        self._control = control
        self._evaluated = False

    def evaluate(self, n_jobs: int = -2) -> "LinkabilityEvaluator":
        """Run the linkability attack.

        Parameters
        ----------
        n_jobs : int, default is -2
            The number of parallel jobs to run for neighbors search.

        Returns
        -------
        self
            The evaluated ``LinkabilityEvaluator`` object.

        """
        self._baseline_links = _random_linkability_attack(
            n_synthetic=self._syn.shape[0], n_attacks=self._n_attacks, n_neighbors=self._n_neighbors
        )

        self._attack_links = _linkability_attack(
            ori=self._ori,
            syn=self._syn,
            n_attacks=self._n_attacks,
            aux_cols=self._aux_cols,
            n_neighbors=self._n_neighbors,
            n_jobs=n_jobs,
        )

        self._control_links = (
            None
            if self._control is None
            else _linkability_attack(
                ori=self._control,
                syn=self._syn,
                n_attacks=self._n_attacks,
                aux_cols=self._aux_cols,
                n_neighbors=self._n_neighbors,
                n_jobs=n_jobs,
            )
        )

        self._evaluated = True
        return self

    def results(self, confidence_level: float = 0.95, n_neighbors: Optional[int] = None) -> EvaluationResults:
        """Raw evaluation results.

        Parameters
        ----------
        confidence_level : float, default is 0.95
            Confidence level for the error bound calculation.
        n_neighbors : int, default is None
            The number of closest neighbors to include in the analysis.
            If `None` (the default), the number used it the one
            given by the constructor. The value of this parameter must
            be smaller of equal to what has been used to initialize this
            evaluator.
        Returns
        -------
        EvaluationResults
            Object containing the success rates for the various attacks.

        """
        if not self._evaluated:
            raise RuntimeError("The linkability evaluator wasn't evaluated yet. Please, run `evaluate()` first.")

        if n_neighbors is None:
            n_neighbors = self._n_neighbors

        if n_neighbors > self._n_neighbors:
            raise ValueError(
                f"Cannot compute linkability results for `n_neighbors` "
                f"({n_neighbors}) larger than value used by constructor "
                f"({self._n_neighbors}. Using `n_neighbors == {self._n_neighbors}`"
            )

        n_control = None if self._control_links is None else self._control_links.count_links(n_neighbors=n_neighbors)

        return EvaluationResults(
            n_attacks=self._n_attacks,
            n_success=self._attack_links.count_links(n_neighbors=n_neighbors),
            n_baseline=self._baseline_links.count_links(n_neighbors=n_neighbors),
            n_control=n_control,
            confidence_level=confidence_level,
        )

    def risk(
        self, confidence_level: float = 0.95, baseline: bool = False, n_neighbors: Optional[int] = None
    ) -> PrivacyRisk:
        """Compute linkability risk.

        The linkability risk reflects how easy linkability attacks are.
        A linkability risk of 1 means that every single attacked record
        could be successfully linked together. A linkability risk of 0
        means that no links were found at all.

        Parameters
        ----------
        confidence_level : float, default is 0.95
            Confidence level for the error bound calculation.
        baseline : bool, default is False
            If True, return the baseline risk computed from a random guessing
            attack. If False (default) return the risk from the real attack.
        n_neighbors : int, default is None
            The number of closest neighbors to include in the analysis.
            If `None` (the default), the number used it the one
            given by the constructor. The value of this parameter must
            be smaller of equal to what has been used to initialize this
            evaluator.

        Returns
        -------
        PrivacyRisk
            Estimate of the linkability risk and its confidence interval.

        """
        results = self.results(confidence_level=confidence_level, n_neighbors=n_neighbors)

        return results.risk(baseline=baseline)
