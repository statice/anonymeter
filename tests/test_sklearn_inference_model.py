# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from anonymeter.evaluators.inference_evaluator import InferenceEvaluator
from anonymeter.evaluators.sklearn_inference_predictor import SklearnInferencePredictor

from tests.fixtures import get_adult


@pytest.mark.parametrize(
    "aux_cols",
    [
        ["type_employer", "capital_loss", "hr_per_week", "age"],
        ["education_num", "type_employer", "capital_loss"],
        ["age", "type_employer", "race"],
    ],
)
@pytest.mark.parametrize("secret", ["capital_gain", "capital_loss"])
def test_inference_evaluator_custom_model_regressor(aux_cols, secret):
    ori = get_adult("ori", n_samples=10)

    # Inference model prep
    categorical_cols = ori[aux_cols].select_dtypes(include=["object"]).columns
    numeric_cols = ori[aux_cols].select_dtypes(include=["number"]).columns

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )
    tree = DecisionTreeRegressor(random_state=42)

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("tree", tree)
    ])
    model.fit(ori[aux_cols], ori[secret])
    inference_model = SklearnInferencePredictor(model)

    # Evaluator
    evaluator = InferenceEvaluator(ori=ori, syn=ori, control=ori, aux_cols=aux_cols, secret=secret, n_attacks=10,
                                   inference_model=inference_model, regression=True)
    evaluator.evaluate(n_jobs=1)
    results = evaluator.results(confidence_level=0)

    np.testing.assert_equal(results.attack_rate, (1, 0))
    np.testing.assert_equal(results.control_rate, (1, 0))
