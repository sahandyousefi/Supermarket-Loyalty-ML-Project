"""Global surrogate: a shallow tree that imitates the forecasts.

A depth-limited regression tree is fitted to the champion's predictions,
not to spend, and its fidelity is the R-squared between surrogate and
champion on members held out from the surrogate fit. A high fidelity means
the printed rules are a faithful summary of how the model forecasts; a low
one means the model is more complex than any short set of rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

from loyalty_spend.features.preprocess import make_preprocessor

Predict = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class Surrogate:
    pipeline: Pipeline
    fidelity: float
    rules: str


def fit_surrogate(
    predict: Predict, X: pd.DataFrame, max_depth: int = 3, seed: int = 0
) -> Surrogate:
    target = predict(X)
    X_fit, X_val, t_fit, t_val = train_test_split(
        X, target, test_size=0.3, random_state=seed
    )
    prep = make_preprocessor("ordinal").set_output(transform="pandas")
    tree = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=20, random_state=seed
    )
    pipe = Pipeline([("prep", prep), ("tree", tree)]).fit(X_fit, t_fit)
    fidelity = float(r2_score(t_val, pipe.predict(X_val)))
    names = list(pipe[:-1].get_feature_names_out())
    rules = export_text(pipe[-1], feature_names=names, decimals=2)
    return Surrogate(pipe, fidelity, rules)
