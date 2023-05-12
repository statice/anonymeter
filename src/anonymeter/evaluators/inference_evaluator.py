# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the inference risk."""
from typing import List, Optional

import numpy as np
import pandas as pd

from anonymeter.neighbors.mixed_types_kneighbors import MixedTypeKNeighbors
from anonymeter.stats.confidence import EvaluationResults, PrivacyRisk


def _run_attack(
    target: pd.DataFrame,
    syn: pd.DataFrame,
    n_attacks: int,
    aux_cols: List[str],
    secret: str,
    n_jobs: int,
    naive: bool,
    regression: Optional[bool],
) -> int:
    if regression is None:
        regression = pd.api.types.is_numeric_dtype(target[secret])

    targets = target.sample(n_attacks, replace=False)

    if naive:
        guesses = syn.sample(n_attacks)[secret]

    else:
        nn = MixedTypeKNeighbors(n_jobs=n_jobs, n_neighbors=1).fit(candidates=syn[aux_cols])

        guesses_idx = nn.kneighbors(queries=targets[aux_cols])
        guesses = syn.iloc[guesses_idx.flatten()][secret]

    return evaluate_inference_guesses(guesses=guesses, secrets=targets[secret], regression=regression).sum()


def evaluate_inference_guesses(
    guesses: pd.Series, secrets: pd.Series, regression: bool, tolerance: float = 0.05
) -> np.ndarray:
    """Evaluate the success of an inference attack.

    The attack is successful if the attacker managed to make a correct guess.

    In case of regression problems, when the secret is a continuous variable,
    the guess is correct if the relative difference between guess and target
    is smaller than a given tolerance. In the case of categorical target
    variables, the inference is correct if the secrets are guessed exactly.

    Parameters
    ----------
    guesses : pd.Series
        Attacker guesses for each of the targets.
    secrets : pd.Series
        Array with the true values of the secret for each of the targets.
    regression : bool
        Whether or not the attacker is trying to solve a classification or
        a regression task. The first case is suitable for categorical or
        discrete secrets, the second for numerical continuous ones.
    tolerance : float, default is 0.05
        Maximum value for the relative difference between target and secret
        for the inference to be considered correct.

    Returns
    -------
    np.array
        Array of boolean values indicating the correcteness of each guess.

    """
    guesses = guesses.values
    secrets = secrets.values

    if regression:
        rel_abs_diff = np.abs(guesses - secrets) / (guesses + 1e-12)
        value_match = rel_abs_diff <= tolerance
    else:
        value_match = guesses == secrets

    nan_match = np.logical_and(pd.isnull(guesses), pd.isnull(secrets))

    return np.logical_or(nan_match, value_match)


class InferenceEvaluator:
    """Privacy evaluator that measures the inference risk.

    The attacker's goal is to use the synthetic dataset to learn about some
    (potentially all) attributes of a target record from the original database.
    The attacker has a partial knowledge of some attributes of the target
    record (the auxiliary information AUX) and uses a similarity score to find
    the synthetic record that matches best the AUX. The success of the attack
    is compared to the baseline scenario of the trivial attacker, who guesses
    at random.

    .. note::
       For a thorough interpretation of the attack result, it is recommended to
       set aside a small portion of the original dataset to use as a *control*
       dataset for the Inference Attack. These control records should **not**
       have been used to generate the synthetic dataset. For good statistical
       accuracy on the attack results, 500 to 1000 control records are usually
       enough.

       Comparing how successful the attack is when targeting the *training* and
       *control* dataset allows for a more sensitive measure of eventual
       information leak during the training process. If, using the synthetic
       data as a base, the attack is more successful against the original
       records in the training set than it is when targeting the control data,
       this indicates that specific information about some records have been
       transferred to the synthetic dataset.

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe with the target records whose secrets the attacker
        will try to guess. This is the private dataframe from which
        the synthetic one has been derived.
    syn : pd.DataFrame
        Dataframe with the synthetic records. It is assumed to be
        fully available to the attacker.
    control : pd.DataFrame (optional)
        Independent sample of original records **not** used to
        create the synthetic dataset. This is used to evaluate
        the excess privacy risk.
    aux_cols : list of str
        Features of the records that are given to the attacker as auxiliary
        information.
    secret : str
        Secret attribute of the targets that is unknown to the attacker.
        This is what the attacker will try to guess.
    regression : bool, optional
        Specifies whether the target of the inference attack is quantitative
        (regression = True) or categorical (regression = False). If None
        (default), the code will try to guess this by checking the type of
        the variable.
    n_attacks : int, default is 500
        Number of attack attempts.

    """

    def __init__(
        self,
        ori: pd.DataFrame,
        syn: pd.DataFrame,
        aux_cols: List[str],
        secret: str,
        regression: Optional[bool] = None,
        n_attacks: int = 500,
        control: Optional[pd.DataFrame] = None,
    ):
        self._ori = ori
        self._syn = syn
        self._control = control
        self._n_attacks = n_attacks
        self._secret = secret
        self._regression = regression
        self._aux_cols = aux_cols
        self._evaluated = False

    def _attack(self, target: pd.DataFrame, naive: bool, n_jobs: int) -> int:
        return _run_attack(
            target=target,
            syn=self._syn,
            n_attacks=self._n_attacks,
            aux_cols=self._aux_cols,
            secret=self._secret,
            n_jobs=n_jobs,
            naive=naive,
            regression=self._regression,
        )

    def evaluate(self, n_jobs: int = -2) -> "InferenceEvaluator":
        r"""Run the inference attack.

        Parameters
        ----------
        n_jobs : int, default is -2
            The number of jobs to run in parallel.

        Returns
        -------
        self
            The evaluated ``InferenceEvaluator`` object.

        """
        self._n_baseline = self._attack(target=self._ori, naive=True, n_jobs=n_jobs)
        self._n_success = self._attack(target=self._ori, naive=False, n_jobs=n_jobs)
        self._n_control = (
            None if self._control is None else self._attack(target=self._control, naive=False, n_jobs=n_jobs)
        )

        self._evaluated = True
        return self

    def results(self, confidence_level: float = 0.95) -> EvaluationResults:
        """Raw evaluation results.

        Parameters
        ----------
        confidence_level : float, default is 0.95
            Confidence level for the error bound calculation.

        Returns
        -------
        EvaluationResults
            Object containing the success rates for the various attacks.

        """
        if not self._evaluated:
            raise RuntimeError("The inference evaluator wasn't evaluated yet. Please, run `evaluate()` first.")

        return EvaluationResults(
            n_attacks=self._n_attacks,
            n_success=self._n_success,
            n_baseline=self._n_baseline,
            n_control=self._n_control,
            confidence_level=confidence_level,
        )

    def risk(self, confidence_level: float = 0.95, baseline: bool = False) -> PrivacyRisk:
        """Compute the inference risk from the success of the attacker.

        This measures how much an attack on training data outperforms
        an attack on control data. An inference risk of 0 means that
        the attack had no advantage on the training data (no inference
        risk), while a value of 1 means that the attack exploited the
        maximally possible advantage.

        Parameters
        ----------
        confidence_level : float, default is 0.95
            Confidence level for the error bound calculation.
        baseline : bool, default is False
            If True, return the baseline risk computed from a random guessing
            attack. If False (default) return the risk from the real attack.

        Returns
        -------
        PrivacyRisk
            Estimate of the inference risk and its confidence interval.

        """
        results = self.results(confidence_level=confidence_level)
        return results.risk(baseline=baseline)
