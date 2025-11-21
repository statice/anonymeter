from typing import Protocol

import pandas as pd

from anonymeter.neighbors.mixed_types_kneighbors import MixedTypeKNeighbors


class InferencePredictor(Protocol):
    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...

    @property
    def sample_targets(self) -> bool:
        ...


class MLModelInference:
    def __init__(self, model, sample_targets: bool = False):
        self._model = model
        self._sample_targets = sample_targets

    @property
    def sample_targets(self) -> bool:
        return self._sample_targets

    def predict(self, x: pd.DataFrame) -> pd.Series:
        return self._model.predict(x)


class KNNInference:
    def __init__(self, data: pd.DataFrame, columns: list[str], target_col: str, n_jobs: int):
        self._nn = MixedTypeKNeighbors(n_jobs=n_jobs, n_neighbors=1).fit(candidates=data[columns])
        self._data = data
        self._target_col = target_col
        self._columns = columns

    @property
    def sample_targets(self) -> bool:
        return True

    def predict(self, x: pd.DataFrame) -> pd.Series:
        guesses_idx = self._nn.kneighbors(queries=x[self._columns])
        if isinstance(guesses_idx, tuple):
            raise RuntimeError("guesses_idx cannot be a tuple")
        return self._data.iloc[guesses_idx.flatten()][self._target_col]