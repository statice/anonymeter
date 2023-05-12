# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the singling out risk."""
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_numeric_dtype
from scipy.optimize import curve_fit

from anonymeter.stats.confidence import EvaluationResults, PrivacyRisk

rng = np.random.default_rng()
logger = logging.getLogger(__name__)


def _escape_quotes(string: str) -> str:
    return string.replace('"', '\\"').replace("'", "\\'")


def _query_expression(col: str, val: Any, dtype: np.dtype) -> str:
    """Generate type-aware query expression."""
    query: str = ""

    if pd.api.types.is_datetime64_any_dtype(dtype):
        f"{col} == '{val}'"
    elif isinstance(val, str):
        query = f"{col} == '{_escape_quotes(val)}'"
    else:
        query = f"{col} == {val}"

    return query


def _query_from_record(record: pd.Series, dtypes: pd.Series, columns: List[str], medians: Optional[pd.Series]) -> str:
    """Construct a query from the attributes in a record."""
    query = []

    for col in columns:

        if pd.isna(record[col]):
            item = ".isna()"
        elif is_bool_dtype(dtypes[col]):
            item = f"== {record[col]}"
        elif is_numeric_dtype(dtypes[col]):

            if medians is None:
                operator = rng.choice([">=", "<="])
            else:
                if record[col] > medians[col]:
                    operator = ">="
                else:
                    operator = "<="
            item = f"{operator} {record[col]}"

        elif is_categorical_dtype(dtypes[col]) and is_numeric_dtype(dtypes[col].categories.dtype):
            item = f"=={record[col]}"
        else:
            if isinstance(record[col], str):
                item = f"== '{_escape_quotes(record[col])}'"
            else:
                item = f'== "{record[col]}"'

        query.append(f"{col}{item}")

    return " & ".join(query)


def _random_operator(data_type: str) -> str:
    if data_type == "categorical":
        ops = ["==", "!="]
    elif data_type == "boolean":
        ops = ["", "not "]
    elif data_type == "numerical":
        ops = ["==", "!=", ">", "<", ">=", "<="]
    else:
        raise ValueError(f"Unknown `data_type`: {data_type}")

    return rng.choice(ops)


def _random_query(unique_values: Dict[str, List[Any]], cols: List[str]):
    """Generate a random query using given columns."""
    query = []

    for col in cols:

        values = unique_values[col]
        val = rng.choice(values)

        if pd.isna(val):
            expression = f"{_random_operator('boolean')}{col}.isna()"
        elif is_bool_dtype(values):
            expression = f"{_random_operator('boolean')}{col}"
        elif is_categorical_dtype(values):
            expression = f"{col} {_random_operator('categorical')} {val}"
        elif is_numeric_dtype(values):
            expression = f"{col} {_random_operator('numerical')} {val}"
        elif isinstance(val, str):
            expression = f"{col} {_random_operator('categorical')} '{_escape_quotes(val)}'"
        else:
            expression = f"{col} {_random_operator('categorical')} '{val}'"

        query.append(expression)

    return " & ".join(query)


def _random_queries(df: pd.DataFrame, n_queries: int, n_cols: int) -> List[str]:

    random_columns = [rng.choice(df.columns, size=n_cols, replace=False).tolist() for _ in range(n_queries)]
    unique_values = {col: df[col].unique() for col in df.columns}

    queries: List[str] = [_random_query(unique_values=unique_values, cols=cols) for cols in random_columns]

    return queries


def safe_query_counts(query: str, df: pd.DataFrame) -> Optional[int]:
    """Return number of elements satisfying a given query."""
    try:
        return len(df.query(query, engine="python"))
    except Exception as ex:
        logger.debug(f"Query {query} failed with {ex}.")
        return None


def singling_out_probability_integral(n: int, w_min: float, w_max: float) -> float:
    """Integral of the singling out probability within a given range.

    The probability that a query singles out in a population of size
    n is defined by the query "weight" (w), i.e. the chance that the
    query matches a random row sampled from the data generating distribution.

    This probability is given by: P(w, n) = n*w * (1 - w)**(n - 1).
    See Cohen and Nissim 2020 [1] for more details.

    References
    ----------
    [1] - https://arxiv.org/abs/1904.06009

    Parameters
    ----------
    n : int
        Size of the population
    w_min : float
        Lower extreme of integration. Must be between 0 and 1.
    w_max : float
        Higher extreme of integration. Must be between w_min and 1.

    Returns
    -------
    float
        The integral of the singling out probability in the given range.

    """
    if w_min < 0 or w_min > 1:
        raise ValueError(f"Parameter `w_min` must be between 0 and 1. Got {w_min} instead.")

    if w_max < w_min or w_max > 1:
        raise ValueError(
            f"Parameter `w_max` must be greater than w_min ({w_min}) and smaller than 1. Got {w_max} instead."
        )

    return ((n * w_min + 1) * (1 - w_min) ** n - (n * w_max + 1) * (1 - w_max) ** n) / (n + 1)


def _measure_queries_success(
    df: pd.DataFrame, queries: List[str], n_repeat: int, n_meas: int
) -> Tuple[np.ndarray, np.ndarray]:
    sizes, successes = [], []
    min_rows = min(1000, len(df))

    for n_rows in np.linspace(min_rows, len(df), n_meas).astype(int):

        for _ in range(n_repeat):
            successes.append(len(_evaluate_queries(df=df.sample(n_rows, replace=False), queries=queries)))
            sizes.append(n_rows)

    return np.array(sizes), np.array(successes)


def _model(x, w_eff, norm):
    return norm * singling_out_probability_integral(n=x, w_min=0, w_max=w_eff)


def _fit_model(sizes: np.ndarray, successes: np.ndarray) -> Callable:
    # initial guesses
    w_eff_guess = 1 / np.max(sizes)
    norm_guess = 1 / singling_out_probability_integral(n=np.max(sizes), w_min=0, w_max=w_eff_guess)

    popt, _ = curve_fit(_model, xdata=sizes, ydata=successes, bounds=(0, (1, np.inf)), p0=(w_eff_guess, norm_guess))

    return lambda x: _model(x, *popt)


def fit_correction_term(df: pd.DataFrame, queries: List[str]) -> Callable:
    """Fit correction for different size of the control dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe on which the queries needs to be evaluated.
    queries : list of strings
        Singling out queries to evaluate on the data.

    Returns
    -------
    callable
        Model of how the number of queries that singles out
        depends on the size of the dataset.

    """
    sizes, successes = _measure_queries_success(df=df, queries=queries, n_repeat=5, n_meas=10)
    return _fit_model(sizes=sizes, successes=successes)


class UniqueSinglingOutQueries:
    """Collection of unique queries that single out in a DataFrame."""

    def __init__(self):
        self._set: Set[str] = set()
        self._list: List[str] = []

    def check_and_append(self, query: str, df: pd.DataFrame):
        """Add a singling out query to the collection.

        A query singles out if the following conditions are met:
            1. single out one record in the dataset.
            2. have either a very low or a very high weight. In
            Both these cases singling out by chance is unlikely.
        Moreover, only queries that are not already in this collection
        can be added.

        Parameters
        ----------
        query : str
            query expression to be added.
        df : pd.DataFrame
            Dataframe on which the queries need to single out.

        """
        sorted_query = "".join(sorted(query))

        if sorted_query not in self._set:

            counts = safe_query_counts(query=query, df=df)

            if counts is not None and counts == 1:
                self._set.add(sorted_query)
                self._list.append(query)

    def __len__(self):
        """Length of the singling out queries in stored."""
        return len(self._list)

    @property
    def queries(self) -> List[str]:
        """Queries that are present in the collection."""
        return self._list


def univariate_singling_out_queries(df: pd.DataFrame, n_queries: int) -> List[str]:
    """Generate singling out queries from rare attributes.

    Parameters
    ----------
    df: pd.DataFrame
            Input dataframe from which queries will be generated.
    n_queries: int
        Number of queries to generate.

    Returns
    -------
    List[str]
        The singling out queries.

    """
    queries = []

    for col in df.columns:

        if df[col].isna().sum() == 1:
            queries.append(f"{col}.isna()")

        if pd.api.types.is_numeric_dtype(df.dtypes[col]):
            values = df[col].dropna().sort_values()

            if len(values) > 0:
                queries.extend([f"{col} <= {values.iloc[0]}", f"{col} >= {values.iloc[-1]}"])

        counts = df[col].value_counts()
        rare_values = counts[counts == 1]

        if len(rare_values) > 0:
            queries.extend([_query_expression(col=col, val=val, dtype=df.dtypes[col]) for val in rare_values.index])

    rng.shuffle(queries)

    so_queries = UniqueSinglingOutQueries()

    for query in queries:
        so_queries.check_and_append(query, df=df)

        if len(so_queries) == n_queries:
            break

    return so_queries.queries


def multivariate_singling_out_queries(df: pd.DataFrame, n_queries: int, n_cols: int) -> List[str]:
    """Generates singling out queries from a combination of attributes.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe from which queries will be generated.
    n_queries: int
        Number of queries to generate.
    n_cols: float
        Number of columns that the attacker uses to create the
        singling out queries.

    Returns
    -------
    List[str]
        The singling out queries.

    """
    so_queries = UniqueSinglingOutQueries()
    medians = df.median(numeric_only=True)

    while len(so_queries) < n_queries:
        record = df.iloc[rng.integers(df.shape[0])]
        columns = rng.choice(df.columns, size=n_cols, replace=False).tolist()

        query = _query_from_record(record=record, dtypes=df.dtypes, columns=columns, medians=medians)

        so_queries.check_and_append(query=query, df=df)

    return so_queries.queries


def _evaluate_queries(df: pd.DataFrame, queries: List[str]) -> List[str]:
    counts = np.array([safe_query_counts(query=q, df=df) for q in queries], dtype=float)

    if np.any(np.isnan(counts)) > 0:
        logger.warning(
            f"Found {np.sum(np.isnan(counts))} failed queries "
            f"out of {len(queries)}. Check DEBUG messages for more details."
        )

    success = counts == 1
    return [q for iq, q in enumerate(queries) if success[iq]]


def _generate_singling_out_queries(df: pd.DataFrame, mode: str, n_attacks: int, n_cols: int) -> List[str]:
    if mode == "univariate":
        queries = univariate_singling_out_queries(df=df, n_queries=n_attacks)

    elif mode == "multivariate":
        queries = multivariate_singling_out_queries(df=df, n_queries=n_attacks, n_cols=n_cols)

    else:
        raise RuntimeError(f"Parameter `mode` can be either `univariate` or `multivariate`. Got {mode} instead.")

    if len(queries) < n_attacks:
        logger.warning(
            f"Attack `{mode}` could generate only {len(queries)} "
            f"singling out queries out of the requested {n_attacks}. "
            "This can probably lead to an underestimate of the "
            "singling out risk."
        )
    return queries


class SinglingOutEvaluator:
    """Privacy evaluator that measures the singling out risk.

    Singling out happens when the attacker can determine that
    there is a single individual in the dataset that has certain
    attributes (for example "zip_code == XXX and first_name == YYY")
    with high enough confidence. According to the Article 29 WGP [2],
    singling out is one of the three risks (together with
    linkability and inference) that a successful anonymization technique
    must protect from.

    See [1] for the definition of some of the concepts used here.

    - [1]: https://arxiv.org/abs/1904.06009
    - [2]: https://ec.europa.eu/justice/article-29/documentation/\
           opinion-recommendation/files/2014/wp216_en.pdf

    Parameters
    ----------
    ori : pd.DataFrame
        Original dataframe on which the success of the singling out attacker
        attacker will be evaluated.
    syn : pd.DataFrame
        Synthetic dataframe used to generate the singling out queries.
    n_attacks : int, default is 500
        Number of singling out attacks to attempt.
    n_cols : int, default is 3
        Number of columns that the attacker uses to create the singling
        out queries.
    control : pd.DataFrame (optional)
        Independent sample of original records **not** used to create the
        synthetic dataset. This is used to evaluate the excess privacy risk.
    """

    def __init__(
        self,
        ori: pd.DataFrame,
        syn: pd.DataFrame,
        n_attacks: int = 500,
        n_cols: int = 3,
        control: Optional[pd.DataFrame] = None,
    ):
        self._ori = ori.drop_duplicates()
        self._syn = syn.drop_duplicates()
        self._n_attacks = n_attacks
        self._n_cols = n_cols
        self._control = None if control is None else control.drop_duplicates()
        self._queries: List[str] = []
        self._random_queries: List[str] = []
        self._evaluated = False

    def queries(self, baseline: bool = False) -> List[str]:
        """Successful singling out queries.

        Parameters
        ----------
        baseline: bool, default is False.
            If True, return the queries used by the baseline attack (i.e.
            created at random). If False (default) return the queries used
            by the "real" attack.

        Returns
        -------
        List[str]:
            successful singling out queries.

        """
        return self._random_queries if baseline else self._queries

    def evaluate(self, mode: str = "multivariate") -> "SinglingOutEvaluator":
        """Run the attack and evaluate the guesses on the original dataset.

        Parameters
        ----------
        mode : str, default is "multivariate"
            Name of the algorithm used to generate the singling out queries.
            Could be either `multivariate` or `univariate`.

        Returns
        -------
        self
            The evaluated singling out evaluator.

        """
        if mode == "multivariate":
            n_cols = self._n_cols
        elif mode == "univariate":
            n_cols = 1
        else:
            raise ValueError(f"mode must be either 'multivariate' or 'univariate', got {mode} instead.")

        baseline_queries = _random_queries(df=self._syn, n_queries=self._n_attacks, n_cols=n_cols)
        self._baseline_queries = _evaluate_queries(df=self._ori, queries=baseline_queries)
        self._n_baseline = len(self._baseline_queries)

        queries = _generate_singling_out_queries(
            df=self._syn, n_attacks=self._n_attacks, n_cols=self._n_cols, mode=mode
        )
        self._queries = _evaluate_queries(df=self._ori, queries=queries)
        self._n_success = len(self._queries)

        if self._control is None:
            self._n_control = None
        else:
            self._n_control = len(_evaluate_queries(df=self._control, queries=queries))

            # correct the number of success against the control set
            # to account for different dataset sizes.
            if len(self._control) != len(self._ori):

                # fit the model to the data:
                fitted_model = fit_correction_term(df=self._control, queries=queries)

                correction = fitted_model(len(self._ori)) / fitted_model(len(self._control))
                self._n_control *= correction

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
            raise RuntimeError("The singling out evaluator wasn't evaluated yet. Please, run `evaluate()` first.")

        return EvaluationResults(
            n_attacks=self._n_attacks,
            n_success=self._n_success,
            n_baseline=self._n_baseline,
            n_control=self._n_control,
            confidence_level=confidence_level,
        )

    def risk(self, confidence_level: float = 0.95, baseline: bool = False) -> PrivacyRisk:
        """Estimate the singling out risk.

        The risk is estimated comparing the number of successfull singling out
        queries to the desired number of attacks (``n_attacks``).

        Parameters
        ----------
        confidence_level : float
            Confidence level for the reported error on the singling out risk.
        baseline : bool, default is False
            If True, return the baseline risk computed from a random guessing
            attack. If False (default) return the risk from the real attack.

        Returns
        -------
        PrivacyRisk
            Estimate of the singling out risk and its confidence interval.

        """
        results = self.results(confidence_level=confidence_level)
        return results.risk(baseline=baseline)
