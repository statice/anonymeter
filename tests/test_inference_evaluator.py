# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest

from anonymeter.evaluators.inference_evaluator import InferenceEvaluator, evaluate_inference_guesses
from tests.fixtures import get_adult


@pytest.mark.parametrize(
    "guesses, secrets, expected",
    [
        (("a", "b"), ("a", "b"), (True, True)),
        ((np.nan, "b"), (np.nan, "b"), (True, True)),
        ((np.nan, np.nan), (np.nan, np.nan), (True, True)),
        ((np.nan, "b"), ("a", np.nan), (False, False)),
        (("a", "b"), ("a", "c"), (True, False)),
        (("b", "b"), ("a", "c"), (False, False)),
        ((1, 0), (2, 0), (False, True)),
    ],
)
def test_evaluate_inference_guesses_classification(guesses, secrets, expected):
    out = evaluate_inference_guesses(guesses=pd.Series(guesses), secrets=pd.Series(secrets), regression=False)
    np.testing.assert_equal(out, expected)


@pytest.mark.parametrize(
    "guesses, secrets, expected",
    [
        ((1.0, 1.0), (1.0, 1.0), (True, True)),
        ((1.01, 1.0), (1.0, 1.01), (True, True)),
        ((1.0, 1.0), (2.0, 1.01), (False, True)),
        ((1.0, 2.0), (2.0, 1.01), (False, False)),
    ],
)
def test_evaluate_inference_guesses_regression(guesses, secrets, expected):
    out = evaluate_inference_guesses(guesses=pd.Series(guesses), secrets=pd.Series(secrets), regression=True)
    np.testing.assert_equal(out, expected)


@pytest.mark.parametrize(
    "guesses, secrets, tolerance, expected",
    [
        ((1.0, 1.0), (1.05, 1.06), 0.05, (True, False)),
        ((1.0, 1.0), (1.05, 1.06), 0.06, (True, True)),
        ((1.0, np.nan), (1.05, np.nan), 0.06, (True, True)),
        ((np.nan, np.nan), (np.nan, np.nan), 0.06, (True, True)),
        ((1, np.nan), (np.nan, 1.06), 0.06, (False, False)),
        ((1.0, 1.0), (1.05, 1.06), 0.04, (False, False)),
        ((1.0, 1.0), (1.25, 1.26), 0.2, (False, False)),
        ((1.0, 1.0), (1.26, 1.25), 0.25, (False, True)),
    ],
)
def test_evaluate_inference_guesses_regression_tolerance(guesses, secrets, tolerance, expected):
    out = evaluate_inference_guesses(
        guesses=pd.Series(guesses), secrets=pd.Series(secrets), tolerance=tolerance, regression=True
    )
    np.testing.assert_equal(out, expected)


@pytest.mark.parametrize(
    "ori, syn, expected",
    [
        ([["a", "b"], ["c", "d"]], [["a", "b"], ["c", "d"]], 1),
        ([["a", "b"], ["c", "d"]], [["a", "b"], ["c", "e"]], 0.5),
        ([["a", "b"], ["c", "d"]], [["a", "h"], ["c", "g"]], 0.0),
    ],
)
def test_inference_evaluator_rates(ori, syn, expected):
    ori = pd.DataFrame(ori, columns=["c0", "c1"])
    syn = pd.DataFrame(syn, columns=["c0", "c1"])
    evaluator = InferenceEvaluator(ori=ori, syn=syn, control=ori, aux_cols=["c0"], secret="c1", n_attacks=2).evaluate(
        n_jobs=1
    )
    results = evaluator.results(confidence_level=0)

    np.testing.assert_equal(results.attack_rate, (expected, 0))
    np.testing.assert_equal(results.control_rate, (expected, 0))


@pytest.mark.parametrize(
    "aux_cols",
    [
        ["type_employer", "capital_loss", "hr_per_week", "age"],
        ["education_num", "marital", "capital_loss"],
        ["age", "type_employer", "race"],
    ],
)
@pytest.mark.parametrize("secret", ["education", "marital", "capital_gain"])
def test_inference_evaluator_leaks(aux_cols, secret):
    ori = get_adult("ori", n_samples=10)
    evaluator = InferenceEvaluator(ori=ori, syn=ori, control=ori, aux_cols=aux_cols, secret=secret, n_attacks=10)
    evaluator.evaluate(n_jobs=1)
    results = evaluator.results(confidence_level=0)

    np.testing.assert_equal(results.attack_rate, (1, 0))
    np.testing.assert_equal(results.control_rate, (1, 0))


def test_evaluator_not_evaluated():
    evaluator = InferenceEvaluator(
        ori=pd.DataFrame(), syn=pd.DataFrame(), control=pd.DataFrame(), aux_cols=[], secret=""
    )
    with pytest.raises(RuntimeError):
        evaluator.risk()
