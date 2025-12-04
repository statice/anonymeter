import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_regressor

from anonymeter.evaluators.inference_predictor import InferencePredictor


class SklearnInferencePredictor(InferencePredictor):
    """Wrapper class to use sklearn methods in the inference evaluator.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A classifier or regressor which implements ::predict().

    """
    def __init__(self, model: BaseEstimator):
        if not (is_classifier(estimator=model) or is_regressor(estimator=model)):
            raise ValueError("Model must be classifier or regressor %s", model)
        if not hasattr(model, "predict"):
            raise ValueError("Model must have a predict method, %s", model)
        self._model = model

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
        prediction = self._model.predict(x)
        if isinstance(prediction, np.ndarray):
            return pd.Series(prediction)
        return prediction
