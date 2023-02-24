# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from anonymeter.evaluators.singling_out_evaluator import (
    SinglingOutEvaluator,
    UniqueSinglingOutQueries,
    multivariate_singling_out_queries,
    safe_query_counts,
    singling_out_probability_integral,
    univariate_singling_out_queries,
)
from tests.fixtures import get_adult


@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
def test_so_general(mode):

    ori = get_adult("ori", n_samples=10)
    syn = get_adult("syn", n_samples=10)
    soe = SinglingOutEvaluator(ori=ori, syn=syn, n_attacks=5).evaluate(mode=mode)

    for q in soe.queries():
        assert len(syn.query(q) == 1)
        assert len(ori.query(q) == 1)


def test_singling_out_queries_unique():

    df = pd.DataFrame({"c1": [1], "c2": [2]})

    queries = UniqueSinglingOutQueries()
    q1, q2 = "c1 == 1", "c2 == 2"

    queries.check_and_append(q1, df=df)
    queries.check_and_append(q1, df=df)
    assert queries.queries == [q1]

    queries.check_and_append(q2, df=df)
    assert queries.queries == [q1, q2]

    queries = UniqueSinglingOutQueries()
    q3, q4 = f"{q1} and {q2}", f"{q2} and {q1}"
    queries.check_and_append(q3, df=df)
    queries.check_and_append(q4, df=df)
    assert queries.queries == [q3]


def test_singling_out_queries():
    df = pd.DataFrame({"c1": [1, 1], "c2": [2, 3]})

    queries = UniqueSinglingOutQueries()
    queries.check_and_append("c1 == 1", df=df)  # does not single out
    assert len(queries) == 0

    queries.check_and_append("c1 == 1 and c2 == 3", df=df)  # does single out
    assert len(queries) == 1


@pytest.mark.parametrize(
    "query, result", [("c1 == 0 and c2 == 'a'", 2), ("c3 == 'fuffa'", None), ("c1 == 2 and c2 == 'c'", 1)]
)
def test_safe_query_counts(query, result):
    df = pd.DataFrame({"c1": [0, 0, 2], "c2": ["a", "a", "c"]})
    assert safe_query_counts(query=query, df=df) == result


def test_univariate_singling_out_queries():
    df = pd.DataFrame({"col1": ["a", "b", "c", "d"]})
    queries = univariate_singling_out_queries(df=df, n_queries=10)
    expected_queries = ["col1 == 'a'", "col1 == 'b'", "col1 == 'c'", "col1 == 'd'"]
    assert sorted(queries) == sorted(expected_queries)


def test_singling_out_query_generator():
    df = pd.DataFrame({"c0": ["a", "b"], "c1": [1.23, 9.87]})
    queries = multivariate_singling_out_queries(df=df, n_queries=2, n_cols=2)
    possible_queries = [
        "c1<= 1.23 & c1>= 9.87",
        "c1<= 1.23 & c0== 'b'",
        "c1<= 1.23 & c0== 'a'",
        "c1>= 9.87 & c1<= 1.23",
        "c1>= 9.87 & c0== 'b'",
        "c1>= 9.87 & c0== 'a'",
        "c0== 'b' & c1<= 1.23",
        "c0== 'b' & c1>= 9.87",
        "c0== 'b' & c0== 'a'",
        "c0== 'a' & c1<= 1.23",
        "c0== 'a' & c1>= 9.87",
        "c0== 'a' & c0== 'b'",
    ]
    for query in queries:
        assert query in possible_queries


@pytest.mark.parametrize("confidence_level", [0.5, 0.68, 0.95, 0.99])
@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
def test_singling_out_risk_estimate(confidence_level, mode):
    ori = get_adult("ori", 10)
    soe = SinglingOutEvaluator(ori=ori, syn=ori, n_attacks=5)
    soe.evaluate(mode=mode)
    risk, ci = soe.risk(confidence_level=confidence_level)
    np.testing.assert_allclose(ci[1], 1.0)


def test_evaluator_not_evaluated():
    soe = SinglingOutEvaluator(ori=pd.DataFrame(), syn=pd.DataFrame())
    with pytest.raises(RuntimeError):
        soe.risk()


@pytest.mark.parametrize("n", [100, 4242, 11235])
@pytest.mark.parametrize("w_min, w_max", [(0, 1), (1 / 10000, 1 / 1000), (0.0013414, 0.2314)])
def test_probability_integral(n, w_min, w_max):
    def _so_probability(n: int, w: float):
        return n * w * ((1 - w) ** (n - 1))

    desired, _ = integrate.quad(lambda x: _so_probability(w=x, n=n), a=w_min, b=w_max)
    integral = singling_out_probability_integral(n=n, w_min=w_min, w_max=w_max)
    np.testing.assert_almost_equal(desired, integral)
