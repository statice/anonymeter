# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest

from anonymeter.evaluators.linkability_evaluator import LinkabilityEvaluator, LinkabilityIndexes
from tests.fixtures import get_adult

rng = np.random.default_rng(seed=42)


@pytest.mark.parametrize("n_attacks", [4, None])
@pytest.mark.parametrize(
    "n_neighbors, confidence_level, expected_risk, expected_ci",
    [
        (1, 0, 0.25, (0.25, 0.25)),
        (2, 0, 1, (1.0, 1.0)),
        (3, 0, 1, (1.0, 1.0)),
        (4, 0, 1, (1.0, 1.0)),
        (1, 0.95, 0.3725, (0.045587, 0.699358)),
        (2, 0.95, 0.7551, (0.5102, 1.0)),
    ],
)
def test_linkability_evaluator(n_neighbors, confidence_level, expected_risk, expected_ci, n_attacks):
    ori = pd.DataFrame({"col0": [0, 0, 4, 0], "col1": [0, 1, 9, 4]})
    syn = pd.DataFrame({"col0": [0, 1, 4, 9], "col1": [0, 1, 4, 9]})

    evaluator = LinkabilityEvaluator(
        ori=ori, syn=syn, n_attacks=n_attacks, n_neighbors=n_neighbors, aux_cols=(["col0"], ["col1"])
    )
    evaluator.evaluate(n_jobs=1)
    risk, ci = evaluator.risk(confidence_level=confidence_level)
    np.testing.assert_allclose(risk, expected_risk, atol=1e-4)
    np.testing.assert_allclose(ci, expected_ci, atol=1e-4)


@pytest.mark.parametrize("n_attacks", [4, None])
@pytest.mark.parametrize(
    "n_neighbors, confidence_level, expected_risk, expected_ci",
    [
        (1, 0, 0.25, (0.25, 0.25)),
        (2, 0, 1, (1.0, 1.0)),
        (3, 0, 1, (1.0, 1.0)),
        (4, 0, 1, (1.0, 1.0)),
        (1, 0.95, 0.3725, (0.045587, 0.699358)),
        (2, 0.95, 0.7551, (0.5102, 1.0)),
    ],
)
def test_linkability_evaluator_neighbors(n_neighbors, confidence_level, expected_risk, expected_ci, n_attacks):
    # see comment in the test_linkability_evaluator to understand
    # the ground truth on which this test is based.
    ori = pd.DataFrame({"col0": [0, 0, 4, 0], "col1": [0, 1, 9, 4]})
    syn = pd.DataFrame({"col0": [0, 1, 4, 9], "col1": [0, 1, 4, 9]})

    evaluator = LinkabilityEvaluator(
        ori=ori, syn=syn, n_attacks=n_attacks, n_neighbors=4, aux_cols=(["col0"], ["col1"])
    )
    evaluator.evaluate(n_jobs=1)
    risk, ci = evaluator.risk(confidence_level=confidence_level, n_neighbors=n_neighbors)
    np.testing.assert_allclose(risk, expected_risk, atol=1e-4)
    np.testing.assert_allclose(ci, expected_ci, atol=1e-4)


@pytest.mark.parametrize("n_neighbors, fails", [(1, False), (2, False), (3, False), (4, False), (5, True), (45, True)])
def test_linkability_evaluator_neighbors_fails(n_neighbors, fails):

    ori = pd.DataFrame({"col0": [0, 0, 4, 0], "col1": [0, 1, 9, 4]})
    syn = pd.DataFrame({"col0": [0, 1, 4, 9], "col1": [0, 1, 4, 9]})

    evaluator = LinkabilityEvaluator(ori=ori, syn=syn, n_attacks=4, n_neighbors=4, aux_cols=(["col0"], ["col1"]))
    evaluator.evaluate(n_jobs=1)

    if fails:
        with pytest.raises(ValueError):
            evaluator.risk(n_neighbors=n_neighbors)
    else:
        evaluator.risk(n_neighbors=n_neighbors)


@pytest.mark.parametrize("n_neighbors, expected_risk", [(1, 0.25), (2, 5 / 6), (3, 1), (4, 1)])
def test_baseline(n_neighbors, expected_risk):
    # note that for the baseline attack, it does not really matter
    # what's inside the synthetic or the original dataframe.
    ori = pd.DataFrame(rng.choice(["a", "b"], size=(400, 2)), columns=["c0", "c1"])
    syn = pd.DataFrame([["a", "a"], ["b", "b"], ["a", "a"], ["a", "a"]], columns=["c0", "c1"])
    evaluator = LinkabilityEvaluator(ori=ori, syn=syn, n_attacks=None, n_neighbors=n_neighbors, aux_cols=("c0", "c1"))
    evaluator.evaluate(n_jobs=1)
    baseline_risk, _ = evaluator.risk(confidence_level=0.95, baseline=True)
    np.testing.assert_allclose(baseline_risk, expected_risk, atol=5e-2)


@pytest.mark.parametrize(
    "n_neighbors, idx_0, idx_1, expected, n_expected",
    [
        (1, [[0], [1], [2], [3]], [[4], [5], [6], [7]], {}, 0),
        (1, [[0], [1], [2], [3]], [[4], [1], [6], [7]], {1: {1}}, 1),
        (1, [[0], [1], [2], [3]], [[4], [1], [6], [7]], {1: {1}}, 1),
        (1, [[0], [1], [6], [3]], [[4], [1], [6], [7]], {1: {1}, 2: {6}}, 2),
        (1, [[0, 1], [2, 3]], [[1, 0], [3, 2]], {}, 0),
        (2, [[0, 1], [2, 3]], [[1, 0], [3, 2]], {0: {0, 1}, 1: {2, 3}}, 2),
    ],
)
def test_find_links(n_neighbors, idx_0, idx_1, expected, n_expected):
    indexes = LinkabilityIndexes(idx_0=np.array(idx_0), idx_1=np.array(idx_1))
    links = indexes.find_links(n_neighbors=n_neighbors)
    n_links = indexes.count_links(n_neighbors=n_neighbors)
    assert links == expected
    assert n_links == n_expected


@pytest.mark.parametrize("confidence_level", [0.5, 0.68, 0.95, 0.99])
def test_linkability_risk(confidence_level):
    ori = get_adult("ori", n_samples=10)
    col_sample = rng.choice(ori.columns, size=4, replace=False)

    evaluator = LinkabilityEvaluator(
        ori=ori, syn=ori, n_attacks=10, n_neighbors=5, aux_cols=(col_sample[:2], col_sample[2:])
    )
    evaluator.evaluate(n_jobs=1)
    risk, ci = evaluator.risk(confidence_level=confidence_level)
    np.testing.assert_allclose(ci[1], 1.0)


def test_evaluator_not_evaluated():
    evaluator = LinkabilityEvaluator(ori=pd.DataFrame(), syn=pd.DataFrame(), aux_cols=[])
    with pytest.raises(RuntimeError):
        evaluator.risk()
