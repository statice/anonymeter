from typing import Protocol

import pandas as pd


class InferencePredictor(Protocol):
    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...
