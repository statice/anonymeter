# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pytest

from anonymeter.stats.confidence import (
    EvaluationResults,
    SuccessRate,
    bind_value,
    probit,
    residual_success,
    success_rate,
)


def test_probit():
    assert np.round(probit(0.95), decimals=2) == 1.96


@pytest.mark.parametrize(
    "n_success, expected_risk, expected_error",
    [
        (850, 0.849, 0.022),
        (0, 0.002, 0.002),
        (1000, 0.998, 0.002),
    ],
)
def test_success_rate(n_success, expected_risk, expected_error):
    rate, error = success_rate(n_total=1000, n_success=n_success, confidence_level=0.95)
    assert np.round(rate, decimals=3) == expected_risk
    assert np.round(error, decimals=3) == expected_error


@pytest.mark.parametrize(
    "attack_rate, control_rate, expected",
    [
        (SuccessRate(0.9, 0.0), SuccessRate(0.8, 0.0), SuccessRate(0.5, 0.0)),
        (SuccessRate(0.9, 0.02), SuccessRate(0.85, 0.02), SuccessRate(0.333, 0.16)),
    ],
)
def test_residual_success(attack_rate, control_rate, expected):
    residual = residual_success(attack_rate=attack_rate, control_rate=control_rate)
    np.testing.assert_equal(np.round(residual, decimals=3), expected)


@pytest.mark.parametrize(
    "point_estimate, error_bound, expected",
    [
        (0.1, 0.3, (0.1, 0.0, 0.4)),
        (1.1, 0.5, (1.0, 0.6, 1.0)),
        (-0.1, 0.2, (0.0, 0.0, 0.1)),
    ],
)
def test_bind_value(point_estimate, error_bound, expected):
    risk = bind_value(point_estimate, error_bound)
    np.testing.assert_almost_equal(np.array([risk.value, risk.ci[0], risk.ci[1]]), expected)


@pytest.mark.parametrize(
    "n_attacks, n_success, n_baseline",
    [(100, 100, 0), (100, 23, 11), (111, 84, 42), (100, 0, 100)],
)
def test_evaluation_results_simple(n_attacks, n_success, n_baseline):
    results = EvaluationResults(
        n_attacks=n_attacks,
        n_success=n_success,
        n_baseline=n_baseline,
        n_control=None,
        confidence_level=0,
    )

    risk = results.risk()
    baseline_risk = results.risk(baseline=True)

    assert results.control_rate is None
    assert results.attack_rate.value == n_success / n_attacks
    assert results.baseline_rate.value == n_baseline / n_attacks

    assert risk.value == n_success / n_attacks
    assert baseline_risk.value == n_baseline / n_attacks
    assert risk.ci == (risk.value, risk.value)
    assert baseline_risk.ci == (baseline_risk.value, baseline_risk.value)


@pytest.mark.parametrize(
    "n_attacks, n_success, n_baseline, n_control, confidence_level, expected_rate, expected_baseline",
    [
        (
            100,
            100,
            0,
            None,
            0.95,
            SuccessRate(value=0.9815032508965071, error=0.01849674910349284),
            SuccessRate(value=0.01849674910349284, error=0.01849674910349284),
        ),
        (
            100,
            100,
            0,
            None,
            0.68,
            SuccessRate(value=0.9951036894831882, error=0.004896310516811869),
            SuccessRate(value=0.0048963105168118685, error=0.004896310516811869),
        ),
        (
            100,
            23,
            11,
            None,
            0.95,
            SuccessRate(value=0.23998824451588613, error=0.08155558571285167),
            SuccessRate(value=0.1244274643007244, error=0.06188550073007873),
        ),
    ],
)
def test_evaluation_results_confidence(
    n_attacks,
    n_success,
    n_baseline,
    n_control,
    confidence_level,
    expected_rate,
    expected_baseline,
):
    results = EvaluationResults(
        n_attacks=n_attacks,
        n_success=n_success,
        n_baseline=n_baseline,
        n_control=n_control,
        confidence_level=confidence_level,
    )
    np.testing.assert_equal(results.attack_rate, expected_rate)
    np.testing.assert_equal(results.baseline_rate, expected_baseline)
    np.testing.assert_equal(results.risk(baseline=False), expected_rate.to_risk())
    np.testing.assert_equal(results.risk(baseline=True), expected_baseline.to_risk())


def test_evaluation_results_warns_baseline():
    with pytest.warns(UserWarning):
        EvaluationResults(
            n_attacks=100,
            n_success=49,
            n_baseline=50,
            n_control=None,
            confidence_level=0.95,
        )


def test_evaluation_results_warns_control():
    with pytest.warns(UserWarning):
        EvaluationResults(n_attacks=100, n_success=49, n_baseline=0, n_control=100, confidence_level=0)


@pytest.mark.parametrize("confidence_level", [-0.1, 1.2])
def test_confidence_exception(confidence_level):
    with pytest.raises(ValueError):
        EvaluationResults(
            n_attacks=100,
            n_success=49,
            n_baseline=0,
            n_control=None,
            confidence_level=confidence_level,
        )
