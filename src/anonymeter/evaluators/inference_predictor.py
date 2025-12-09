# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""A protocol for a custom inference predictor."""
from typing import Protocol

import pandas as pd


class InferencePredictor(Protocol):
    """Interface for custom inference models.

    It is used as `inference_model` in the InferenceEvaluator in inference_evaluator.py.

    For an example usage refer to the SklearnInferencePredictor in sklearn_inference_predictor.py.
    """
    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict the targets for input `x`.

        Parameters
        ----------
        x : pd.DataFrame
            The input data to predict.

        Returns
        -------
        pd.Series
            The predictions as pd.Series.

        """
        ...
