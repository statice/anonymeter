# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Functions for estimating rates and errors in privacy attacks."""

import warnings
from math import sqrt
from typing import NamedTuple, Optional, Tuple

from scipy.stats import norm


class PrivacyRisk(NamedTuple):
    """Measure of a privacy risk.

    Parameters
    ----------
    value : float
        Best estimate of the privacy risk.
    ci : (float, float)
        Confidence interval on the best estimate.

    """

    value: float
    ci: Tuple[float, float]


class SuccessRate(NamedTuple):
    """Estimate of the success rate of a privacy attack.

    Parameters
    ----------
    value : float
        Best estimate of the success rate of the attacker.
    error : float
        Error on the best estimate.

    """

    value: float
    error: float

    def to_risk(self) -> PrivacyRisk:
        """Convert attacker success rate to `PrivacyRisk`."""
        return bind_value(point_estimate=self.value, error_bound=self.error)


def probit(confidence_level: float) -> float:
    """Compute the probit for the given confidence level."""
    return norm.ppf(0.5 * (1.0 + confidence_level))


def success_rate(n_total: int, n_success: int, confidence_level: float) -> SuccessRate:
    """Estimate success rate in a Bernoulli-distributed sample.

    Attack scores follow a Bernoulli distribution (success/failure with rates p/1-p).
    The Wilson score interval is a frequentist-type estimator for success rate and
    confidence which is robust in problematic cases (e.g., when p goes extreme or
    sample size is small). The estimated rate is a weighted average between the
    MLE result and 0.5 which, however, in the sample sizes used in privacy attacks
    does not differ visibly from the MLE outcome.

    Parameters
    ----------
    n_total : int
        Size of the sample.
    n_success : int
        Number of successful trials in the sample.
    confidence_level : float
        Confidence level for the error estimation.

    Returns
    -------
    float
        Point estimate for the success rate.
    float
        Error bound of the point-estimated rate for the requested confidence level.

    Notes
    -----
    E.B. WILSON
    Probable inference, the law of succession, and statistical inference
    Journal of the American Statistical Association 22, 209-212 (1927)
    DOI 10.1080/01621459.1927.10502953

    """
    if confidence_level > 1 or confidence_level < 0:
        raise ValueError(f"Parameter `confidence_level` must be between 0 and 1. Got {confidence_level} instead.")

    z = probit(confidence_level)

    z_squared = z * z
    n_success_var = n_success * (n_total - n_success) / n_total
    denominator = n_total + z_squared

    rate = (n_success + 0.5 * z_squared) / denominator
    error = (z / denominator) * sqrt(n_success_var + 0.25 * z_squared)
    return SuccessRate(value=rate, error=error)


def residual_success(
    attack_rate: SuccessRate,
    control_rate: SuccessRate,
) -> SuccessRate:
    """Compute residual success in a privacy attack.

    Residual success is defined as the excess of training attack
    success over control attack success, normalized w.r.t.
    the margin of improvement (unsuccessful attacks on control).

    Parameters
    ----------
    attack_rate : SuccessRate
        Success rate on training data.
    control_rate : SuccessRate
        Success rate on control data.

    Returns
    -------
    SuccessRate
        Residual success score without sign correction (i.e., negative
        outcome if control more attack-able than training). The correction
        would yield ``0 ≤ score ≤ 1`` (zero for negative uncorrected score).
        The error estimate is the propagated error bound of the residual
        success rate.

    """
    residual = (attack_rate.value - control_rate.value) / (1.0 - control_rate.value)

    # propagate the error using
    # dF = sqrt[ (dF/dx)^2 dx^2 + (dF/dy)^2 dy^2 + ... ]
    der_wrt_attack = 1 / abs(1 - control_rate.value)
    der_wrt_control = (attack_rate.value - 1) / (1 - control_rate.value) ** 2

    error = sqrt((attack_rate.error * der_wrt_attack) ** 2 + (control_rate.error * der_wrt_control) ** 2)

    return SuccessRate(value=residual, error=error)


def bind_value(point_estimate: float, error_bound: float) -> PrivacyRisk:
    """Force point_estimate and error into fixed bounds.

    Parameters
    ----------
    point_estimate : float
        Point estimate of a rate or risk value.
    error_bound : float
        Symmetric error around the point estimate.


    Returns
    -------
    float
        Point estimate respecting the bounds 0–1 or 0–100.
    Tuple[float, float]
        Asymmetric confidence interval respecting the bounds 0–1 or 0–100.

    """
    bound_point = min(max(point_estimate, 0.0), 1.0)
    bound_lower = min(max(point_estimate - error_bound, 0.0), 1.0)
    bound_upper = min(max(point_estimate + error_bound, 0.0), 1.0)
    return PrivacyRisk(value=bound_point, ci=(bound_lower, bound_upper))


class EvaluationResults:
    """Results of a privacy evaluator.

    This class will compute the attacker's success rates
    and estimate for the corresponding privacy risk.

    Parameters
    ----------
    n_attacks : int
        Total number of attacks performed.
    n_success : int
        Number of successful attacks.
    n_baseline : int
        Number of successful attacks for the
        baseline (i.e. random-guessing) attacker.
    n_control : int, default is None
        Number of successful attacks against the
        control dataset. If this parameter is not None
        the privacy risk will be measured relative to
        the attacker success on the control set.
    confidence_level : float, default is 0.95
        Desired confidence level for the confidence
        intervals on the risk.

    """

    def __init__(
        self,
        n_attacks: int,
        n_success: int,
        n_baseline: int,
        n_control: Optional[int] = None,
        confidence_level: float = 0.95,
    ):
        self.attack_rate = success_rate(n_total=n_attacks, n_success=n_success, confidence_level=confidence_level)

        self.baseline_rate = success_rate(n_total=n_attacks, n_success=n_baseline, confidence_level=confidence_level)

        self.control_rate = (
            None
            if n_control is None
            else success_rate(n_total=n_attacks, n_success=n_control, confidence_level=confidence_level)
        )

        self.n_attacks = n_attacks
        self.n_success = n_success
        self.n_baseline = n_baseline
        self.n_control = n_control

        self._sanity_check()

    def _sanity_check(self):
        if self.baseline_rate.value >= self.attack_rate.value:
            warnings.warn(
                "Attack is as good or worse as baseline model. "
                f"Estimated rates: attack = {self.attack_rate.value}, "
                f"baseline = {self.baseline_rate.value}. "
                "Analysis results cannot be trusted.",
                stacklevel=2,
            )

        if self.control_rate is not None and self.control_rate.value == 1:
            warnings.warn("Success of control attack is 100%. Cannot measure residual privacy risk.", stacklevel=2)

    def risk(self, baseline: bool = False) -> PrivacyRisk:
        """Estimate the privacy risk."""
        if baseline:
            return self.baseline_rate.to_risk()

        if self.control_rate is None:
            return self.attack_rate.to_risk()
        else:
            return residual_success(attack_rate=self.attack_rate, control_rate=self.control_rate).to_risk()
