# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the singling out risk."""

import logging
import operator
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from scipy.optimize import curve_fit

from anonymeter.stats.confidence import EvaluationResults, PrivacyRisk

logger = logging.getLogger(__name__)


def _escape_quotes(string: str) -> str:
    return string.replace('"', '\\"').replace("'", "\\'")


def _query_from_record(
    record: dict,
    dtypes: dict,  # map col -> pl.DataType
    columns: List[str],
    medians: dict,  # map col -> median value
    rng: np.random.Generator,
) -> pl.Expr:
    """Construct a query from the attributes in a record."""
    expr_components = []

    for col in sorted(columns):
        val = record[col]

        if val is None or (isinstance(val, float) and np.isnan(val)):
            expr_col = pl.col(col).is_null()

        elif dtypes[col] == pl.Boolean or dtypes[col] == pl.Categorical:
            expr_col = pl.col(col) == val

        elif dtypes[col].is_numeric():
            if medians is None:
                op = _operator_choice([operator.ge, operator.le], rng)
            else:
                # query for more extreme values to increase the chances of singling out
                if val > medians[col]:
                    op = operator.ge
                else:
                    op = operator.le

            expr_col = op(pl.col(col), val)

        else:
            if isinstance(val, str):
                expr_col = pl.col(col) == _escape_quotes(val)
            else:
                expr_col = pl.col(col) == str(val)

        expr_components.append(expr_col)

    expr = reduce(operator.and_, expr_components)
    return expr

def _operator_choice(
    operators: Sequence[Callable[[Any, Any], bool]],
    rng: np.random.Generator
) -> Callable[[Any, Any], bool]:
    return rng.choice(operators) #type: ignore[arg-type] # signature of "choice" does not accept a list of callables but works fine in practice

def _random_operator(
    data_type: str, rng: np.random.Generator
) -> Callable[[Any, Any], Union[bool, pl.Expr]]:
    if data_type in ["categorical", "boolean"]:
        ops: Sequence[Callable[[Any, Any], bool]] = [operator.eq, operator.ne]
    elif data_type == "numerical":
        ops = [
            operator.eq,
            operator.ne,
            operator.gt,
            operator.lt,
            operator.ge,
            operator.le,
        ]
    else:
        raise ValueError(f"Unknown `data_type`: {data_type}")

    return _operator_choice(ops, rng)


def _random_query(
    unique_values: Dict[str, List[Any]],
    cols: List[str],
    column_types: Dict[str, str],
    rng: np.random.Generator,
) -> pl.Expr:
    exprs = []
    for col in sorted(cols):
        values = unique_values[col]
        val = rng.choice(values)

        data_type = column_types[col]
        op = _random_operator(data_type, rng)

        if val is None:
            # Null checks
            if op == operator.eq:
                e = pl.col(col).is_null()
            else:
                e = ~pl.col(col).is_null()
        elif data_type == "boolean":
            if op == operator.eq:
                e = pl.col(col)
            else:
                e = ~pl.col(col)
        else:
            e = cast(pl.Expr, op(pl.col(col), val))

        exprs.append(e)

    # use bitwise and
    return reduce(operator.and_, exprs)


def _convert_polars_dtype(dtype: pl.DataType) -> str:
    if dtype in (pl.Boolean,):
        return "boolean"
    elif dtype.is_numeric():
        return "numerical"
    elif dtype in (pl.Utf8, pl.Categorical):
        return "categorical"
    return "categorical"  # Fallback


def _random_queries(
    df: pl.DataFrame,
    n_queries: int,
    n_cols: int,
    rng: np.random.Generator,
) -> List[pl.Expr]:
    unique_values = {col: df[col].unique().to_list() for col in df.columns}
    column_types = {
        col: _convert_polars_dtype(df[col].dtype)
        for col in df.columns
    }

    queries = []
    for _ in range(n_queries):
        selected_cols = rng.choice(
            df.columns, size=n_cols, replace=False
        ).tolist()

        queries.append(
            _random_query(
                unique_values=unique_values,
                cols=selected_cols,
                column_types=column_types,
                rng=rng
            )
        )

    return queries


def singling_out_probability_integral(
    n: int, w_min: float, w_max: float
) -> float:
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
        raise ValueError(
            f"Parameter `w_min` must be between 0 and 1. Got {w_min} instead."
        )

    if w_max < w_min or w_max > 1:
        raise ValueError(
            f"Parameter `w_max` must be greater than w_min ({w_min}) and smaller than 1. Got {w_max} instead."
        )

    return (
        (n * w_min + 1) * (1 - w_min) ** n - (n * w_max + 1) * (1 - w_max) ** n
    ) / (n + 1)


def _measure_queries_success(
    df: pl.DataFrame, queries: List[pl.Expr], n_repeat: int, n_meas: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    sizes, successes = [], []
    min_rows = min(1000, len(df))

    for n_rows in np.linspace(min_rows, len(df), n_meas).astype(int):
        for _ in range(n_repeat):
            successes.append(len(_evaluate_queries(df=df.sample(n_rows, with_replacement=False), queries=queries)))
            sizes.append(n_rows)

    return np.array(sizes), np.array(successes)


def _model(x, w_eff, norm):
    return norm * singling_out_probability_integral(n=x, w_min=0, w_max=w_eff)


def _fit_model(sizes: npt.NDArray, successes: npt.NDArray) -> Callable:
    # initial guesses
    w_eff_guess = 1 / np.max(sizes)
    norm_guess = 1 / singling_out_probability_integral(
        n=np.max(sizes), w_min=0, w_max=w_eff_guess
    )

    popt, _ = curve_fit(
        _model,
        xdata=sizes,
        ydata=successes,
        bounds=(0, (1, np.inf)),
        p0=(w_eff_guess, norm_guess),
    )

    return lambda x: _model(x, *popt)


def fit_correction_term(df: pl.DataFrame, queries: List[pl.Expr]) -> Callable:
    """Fit correction for different size of the control dataset.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe on which the queries needs to be evaluated.
    queries : list of polars expressions
        Singling out queries to evaluate on the data.

    Returns
    -------
    callable
        Model of how the number of queries that singles out
        depends on the size of the dataset.

    """
    sizes, successes = _measure_queries_success(
        df=df, queries=queries, n_repeat=5, n_meas=10
    )
    return _fit_model(sizes=sizes, successes=successes)


class UniqueSinglingOutQueries:
    """Collection of unique queries that single out in a DataFrame.

    Parameters
    ----------
    max_size : Optional[int]
        Maximum number of singling out queries to store in this collection.
    """

    def __init__(self, max_size: Optional[int] = None):
        self._set: Set[str] = set()
        self._list: List[pl.Expr] = []
        self._max_size: Optional[int] = max_size

    def check_and_extend(self, queries: List[pl.Expr], df: pl.DataFrame):
        """Add singling-out queries to the collection.

        Only queries that are not already in this collection can be added.
        Maximum number of queries can be limited.

        Parameters
        ----------
        queries : List[pl.Expr]
            List of potentially singling-out queries.
        df : pl.DataFrame
            Dataframe on which the queries need to single out.

        """
        if self._max_size and len(self._list) >= self._max_size:
            return

        counts = _evaluate_queries(df=df, queries=queries)

        for query, count in zip(queries, counts):
            if count == 1:
                query_str = str(query)
                if query_str not in self._set:
                    self._set.add(query_str)
                    self._list.append(query)
                    if self._max_size and len(self._list) >= self._max_size:
                        return

    def __len__(self):
        """Length of the singling out queries in stored."""
        return len(self._list)

    @property
    def queries(self) -> List[pl.Expr]:
        """Queries that are present in the collection."""
        return self._list


def univariate_singling_out_queries(
    df: pl.DataFrame, n_queries: int, rng: np.random.Generator
) -> List[pl.Expr]:
    """Generate singling out queries from rare attributes.

    Parameters
    ----------
    df: pd.DataFrame
            Input dataframe from which queries will be generated.
    n_queries: int
        Number of queries to generate.
    rng: np.random.Generator
        Random number generator used when generating the queries.

    Returns
    -------
    List[pl.Expr]
        The singling out queries.

    """
    queries = []

    schema = df.schema

    for col in df.columns:
        # Exactly one null
        null_count = df.select(pl.col(col).is_null().sum()).item()
        if null_count == 1:
            queries.append(pl.col(col).is_null())

        # Numeric columns
        if schema[col].is_numeric():
            non_null_count = df[col].drop_nulls().len()

            if non_null_count > 0:
                col_min = df[col].min()
                col_max = df[col].max()
                queries.extend(
                    [
                        pl.col(col) <= col_min,
                        pl.col(col) >= col_max,
                    ]
                )

        # Rare values
        counts_df = df.group_by(col).len()
        rare_values_df = counts_df.filter(pl.col("len") == 1)
        rare_values = rare_values_df.select(col).to_series().to_list()
        if len(rare_values) > 0:
            queries.extend([pl.col(col) == val for val in rare_values])

    rng.shuffle(queries) #type: ignore[arg-type] # signature of "shuffle" does not accept a list of expressions but works fine in practice

    unique_so_queries = UniqueSinglingOutQueries(max_size=n_queries)
    unique_so_queries.check_and_extend(queries, df)

    return unique_so_queries.queries


def multivariate_singling_out_queries(
    df: pl.DataFrame,
    n_queries: int,
    n_cols: int,
    max_attempts: Optional[int],
    rng: np.random.Generator,
    batch_size: int = 1000,
) -> List[pl.Expr]:
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
    max_attemps: int, optional.
        Maximum number of attempts that the attacker can make to generate
        the requested ``n_attacks`` singling out queries. This is useful to
        avoid excessively long running calculations. There can be combinations
        of hyperparameters (`n_cols`) and datasets that make the task of
        generating enough singling out queries is too hard. This parameter
        caps the total number of query generation attempts, both those that
        are successfull as those that are not. If ``max_attempts`` is None,
        no limit will be imposed.
    rng: np.random.Generator
        Random number generator used when generating the queries.
    batch_size: int, default is 1000
        Number of queries to generate in a batch. Evaluation in batches
        substantially speeds up the process of generating queries.


    Returns
    -------
    List[pl.Expr]
        The singling out queries.

    """
    unique_so_queries = UniqueSinglingOutQueries(max_size=n_queries)

    medians = df.median()
    medians_dict = medians.to_dicts()[0]
    dtypes_dict = {col: df[col].dtype for col in df.columns}

    n_attempts = 0
    if max_attempts is not None and batch_size > max_attempts:
        batch_size = max_attempts

    while len(unique_so_queries) < n_queries:
        if max_attempts is not None and n_attempts >= max_attempts:
            logger.warning(
                f"Reached maximum number of attempts {max_attempts} when generating singling out queries. "
                f"Returning {len(unique_so_queries)} instead of the requested {n_queries}."
            )
            return unique_so_queries.queries

        # Generate a batch of queries

        # Pre-sample all random row indices
        random_indices = rng.integers(
            low=0, high=df.shape[0], size=batch_size
        )

        # Extract all records in bulk
        records = df[random_indices].to_dicts()

        # Pre-sample all column choices
        selected_columns = [
            rng.choice(df.columns, size=n_cols, replace=False).tolist()
            for _ in range(batch_size)
        ]

        queries_batch = [
            _query_from_record(
                record=record,
                dtypes=dtypes_dict,
                columns=columns,
                medians=medians_dict,
                rng=rng,
            )
            for record, columns in zip(records, selected_columns)
        ]

        # Store queries that single out and that haven't been seen before
        unique_so_queries.check_and_extend(queries_batch, df)

        n_attempts += batch_size
        if len(unique_so_queries) >= n_queries:
            break

    return unique_so_queries.queries


def _evaluate_queries(
    df: pl.DataFrame, queries: List[pl.Expr]
) -> Tuple[int, ...]:
    if len(queries) == 0:
        return ()

    result_df = df.select(
        [
            q.cast(pl.Int64).sum().alias(f"count_{i}")
            for i, q in enumerate(queries)
        ]
    )
    counts = result_df.row(0)
    return counts


def _evaluate_queries_and_return_successful(
    df: pl.DataFrame, queries: List[pl.Expr]
) -> List[pl.Expr]:
    counts = _evaluate_queries(df=df, queries=queries)

    counts_np = np.array(counts, dtype=float)

    if np.any(np.isnan(counts_np)) > 0:
        logger.warning(
            f"Found {np.sum(np.isnan(counts_np))} failed queries "
            f"out of {len(queries)}. Check DEBUG messages for more details."
        )

    success = counts_np == 1
    return [q for iq, q in enumerate(queries) if success[iq]]


def _generate_singling_out_queries(
    df: pl.DataFrame,
    mode: str,
    n_attacks: int,
    n_cols: int,
    max_attempts: Optional[int],
    rng: np.random.Generator,
) -> List[pl.Expr]:
    if mode == "univariate":
        queries = univariate_singling_out_queries(
            df=df, n_queries=n_attacks, rng=rng
        )

    elif mode == "multivariate":
        queries = multivariate_singling_out_queries(
            df=df,
            n_queries=n_attacks,
            n_cols=n_cols,
            max_attempts=max_attempts,
            rng=rng,
        )

    else:
        raise RuntimeError(
            f"Parameter `mode` can be either `univariate` or `multivariate`. Got {mode} instead."
        )

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
    max_attempts : int or None, default is 10.000.000
        Maximum number of attempts that the attacker can make to generate
        the requested ``n_attacks`` singling out queries. This is useful to
        avoid excessively long running calculations. There can be combinations
        of hyperparameters (`n_cols`) and datasets that make the task of
        generating enough singling out queries is too hard. This parameter
        caps the total number of query generation attempts, both those that
        are successfull as those that are not. If ``max_attempts`` is None,
        no limit will be imposed.
    seed : int or None, default is None
        Random seed used to generate the singling out queries.

    """

    def __init__(
        self,
        ori: pd.DataFrame,
        syn: pd.DataFrame,
        n_attacks: int = 500,
        n_cols: int = 3,
        control: Optional[pd.DataFrame] = None,
        max_attempts: Optional[int] = 10000000,
        seed: Optional[int] = None,
    ):
        ori = pl.DataFrame(ori)
        syn = pl.DataFrame(syn)
        self._ori = ori.unique(maintain_order=True)
        self._syn = syn.unique(maintain_order=True)
        self._n_attacks = n_attacks
        self._n_cols = n_cols
        if control is None:
            self._control = None
        else:
            control = pl.DataFrame(control)
            self._control = control.unique(maintain_order=True)
        self._max_attempts = max_attempts
        self._queries: List[pl.Expr] = []
        self._random_queries: List[pl.Expr] = []
        self._evaluated = False
        self._rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

    def queries(self, baseline: bool = False) -> List[pl.Expr]:
        """Successful singling out queries.

        Parameters
        ----------
        baseline: bool, default is False.
            If True, return the queries used by the baseline attack (i.e.
            created at random). If False (default) return the queries used
            by the "real" attack.

        Returns
        -------
        List[pl.Expr]:
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
            raise ValueError(
                f"mode must be either 'multivariate' or 'univariate', got {mode} instead."
            )

        queries = _generate_singling_out_queries(
            df=self._syn,
            n_attacks=self._n_attacks,
            n_cols=self._n_cols,
            mode=mode,
            max_attempts=self._max_attempts,
            rng=self._rng,
        )
        self._queries = _evaluate_queries_and_return_successful(
            df=self._ori, queries=queries
        )
        self._n_success = len(self._queries)

        baseline_queries = _random_queries(
            df=self._syn,
            n_queries=self._n_attacks,
            n_cols=n_cols,
            rng=self._rng,
        )
        self._baseline_queries = _evaluate_queries_and_return_successful(
            df=self._ori, queries=baseline_queries
        )
        self._n_baseline = len(self._baseline_queries)

        if self._control is None:
            self._n_control = None
        else:
            self._n_control = len(
                _evaluate_queries_and_return_successful(
                    df=self._control, queries=queries
                )
            )

            # correct the number of success against the control set
            # to account for different dataset sizes.
            if len(self._control) != len(self._ori):
                # fit the model to the data:
                fitted_model = fit_correction_term(
                    df=self._control, queries=queries
                )

                correction = fitted_model(len(self._ori)) / fitted_model(
                    len(self._control)
                )
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
            raise RuntimeError(
                "The singling out evaluator wasn't evaluated yet. Please, run `evaluate()` first."
            )

        return EvaluationResults(
            n_attacks=self._n_attacks,
            n_success=self._n_success,
            n_baseline=self._n_baseline,
            n_control=self._n_control,
            confidence_level=confidence_level,
        )

    def risk(
        self, confidence_level: float = 0.95, baseline: bool = False
    ) -> PrivacyRisk:
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
