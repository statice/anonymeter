# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import integrate

from anonymeter.evaluators.singling_out_evaluator import (
    SinglingOutEvaluator,
    UniqueSinglingOutQueries,
    multivariate_singling_out_queries,
    _evaluate_queries,
    singling_out_probability_integral,
    univariate_singling_out_queries,
)

from tests.fixtures import get_adult


@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
def test_so_general(mode):
    ori = get_adult("ori", n_samples=10)
    syn = get_adult("syn", n_samples=10)
    soe = SinglingOutEvaluator(ori=ori, syn=syn, n_attacks=5).evaluate(
        mode=mode
    )

    for q in soe.queries():
        print(q)
        assert len(soe._syn.filter(q)) == 1
        assert len(soe._ori.filter(q)) == 1


def test_singling_out_queries_unique():
    df = pl.DataFrame({"c1": [1], "c2": [2]})

    queries = UniqueSinglingOutQueries(max_size=2)
    q1 = pl.col("c1") == 1
    q2 = pl.col("c2") == 2

    queries.check_and_extend([q1], _evaluate_queries(df=df, queries=[q1]))
    queries.check_and_extend([q1], _evaluate_queries(df=df, queries=[q1]))
    assert [str(q) for q in queries.queries] == [str(q1)]

    queries.check_and_extend([q2], _evaluate_queries(df=df, queries=[q2]))
    assert [str(q) for q in queries.queries] == [str(q1), str(q2)]


def test_singling_out_queries_same_characters():
    df = pl.DataFrame([{"c": 1.2}, {"c": 2.1}])

    queries = UniqueSinglingOutQueries(max_size=2)
    q1 = pl.col("c") == 1.2
    q2 = pl.col("c") == 2.1

    queries.check_and_extend([q1], _evaluate_queries(df=df, queries=[q1]))
    queries.check_and_extend([q1], _evaluate_queries(df=df, queries=[q1]))
    assert queries.queries == [q1]

    queries.check_and_extend([q2], _evaluate_queries(df=df, queries=[q2]))
    assert queries.queries == [q1, q2]


def test_singling_out_queries():
    df = pl.DataFrame({"c1": [1, 1], "c2": [2, 3]})

    queries = UniqueSinglingOutQueries(max_size=2)
    q1 = pl.col("c1") == 1  # does NOT single out
    queries.check_and_extend([q1], _evaluate_queries(df=df, queries=[q1]))
    assert len(queries) == 0

    q2 = (pl.col("c1") == 1) & (pl.col("c2") == 3)  # DOES single out

    queries.check_and_extend([q2], _evaluate_queries(df=df, queries=[q2]))
    assert len(queries) == 1


@pytest.mark.parametrize(
    "query, result",
    [
        ((pl.col("c1") == 0) & (pl.col("c2") == "a"), 2),
        (
            pl.col("c3") == "fuffa",
            None,
        ),  # missing column => _evaluate_queries returns None
        ((pl.col("c1") == 2) & (pl.col("c2") == "c"), 1),
    ],
)
def test_evaluate_queries(query, result):
    df = pl.DataFrame({"c1": [0, 0, 2], "c2": ["a", "a", "c"]})
    out = _evaluate_queries(df=df, queries=[query])
    if result is None:
        assert out is None
    else:
        assert out[0] == result


def test_univariate_singling_out_queries():
    df = pl.DataFrame({"col1": ["a", "b", "c", "d"]})
    queries = univariate_singling_out_queries(df=df, n_queries=10)
    expected_queries = [str(pl.col("col1") == v) for v in ["a", "b", "c", "d"]]
    assert sorted(map(str, queries)) == sorted(expected_queries)


def test_singling_out_query_generator():
    df = pl.DataFrame({"c0": ["a", "b"], "c1": [1.23, 9.87]})
    queries = multivariate_singling_out_queries(
        df=df, n_queries=2, n_cols=2, max_attempts=None
    )
    expected_exprs = [
        (pl.col("c1") <= 1.23) & (pl.col("c1") >= 9.87),
        (pl.col("c1") >= 9.87) & (pl.col("c1") <= 1.23),
        (pl.col("c0") == "b") & (pl.col("c1") <= 1.23),
        (pl.col("c0") == "b") & (pl.col("c1") >= 9.87),
        (pl.col("c0") == "b") & (pl.col("c0") == "a"),
        (pl.col("c0") == "a") & (pl.col("c1") <= 1.23),
        (pl.col("c0") == "a") & (pl.col("c1") >= 9.87),
        (pl.col("c0") == "a") & (pl.col("c0") == "b"),
    ]
    expected_strings = {str(e) for e in expected_exprs}

    for query in queries:
        assert str(query) in expected_strings


@pytest.mark.parametrize("confidence_level", [0.5, 0.68, 0.95, 0.99])
@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
def test_singling_out_risk_estimate(confidence_level, mode):
    ori = get_adult("ori", 10)
    soe = SinglingOutEvaluator(ori=ori, syn=ori, n_attacks=5)
    soe.evaluate(mode=mode)
    _, ci = soe.risk(confidence_level=confidence_level)
    np.testing.assert_allclose(ci[1], 1.0)


def test_evaluator_not_evaluated():
    soe = SinglingOutEvaluator(ori=pd.DataFrame(), syn=pd.DataFrame())
    with pytest.raises(RuntimeError):
        soe.risk()


@pytest.mark.parametrize("n", [100, 4242, 11235])
@pytest.mark.parametrize(
    "w_min, w_max", [(0, 1), (1 / 10000, 1 / 1000), (0.0013414, 0.2314)]
)
def test_probability_integral(n, w_min, w_max):
    def _so_probability(n: int, w: float):
        return n * w * ((1 - w) ** (n - 1))

    desired, _ = integrate.quad(
        lambda x: _so_probability(w=x, n=n), a=w_min, b=w_max
    )
    integral = singling_out_probability_integral(n=n, w_min=w_min, w_max=w_max)
    np.testing.assert_almost_equal(desired, integral)


@pytest.mark.parametrize("max_attempts", [1, 2, 3])
def test_so_evaluator_max_attempts(max_attempts):
    ori = get_adult("ori", 10)
    soe = SinglingOutEvaluator(
        ori=ori, syn=ori, n_attacks=10, max_attempts=max_attempts
    )
    soe.evaluate(mode="multivariate")

    assert len(soe.queries()) <= max_attempts
