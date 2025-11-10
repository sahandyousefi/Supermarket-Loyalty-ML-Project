"""Propensity scores estimated out of fold.

Every unit's propensity is predicted by a model that did not see that unit.
In-sample propensities from a flexible model are pushed towards 0 and 1,
which inflates inverse-probability weights and breaks the guarantees of the
doubly robust estimators (Chernozhukov et al., 2018).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loyalty_spend.arrays import FloatArray


def design(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot categories and standardized numerics of any covariate frame."""
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("num", StandardScaler(), numeric),
        ]
    )


def propensity_model(
    X: pd.DataFrame, kind: str = "logistic", seed: int = 0
) -> Pipeline:
    """Logistic regression or gradient boosting on the covariate design."""
    if kind == "logistic":
        clf: BaseEstimator = LogisticRegression(C=1.0, max_iter=2000)
    elif kind == "gbm":
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            random_state=seed,
        )
    else:
        raise ValueError("kind must be 'logistic' or 'gbm'")
    return Pipeline([("design", design(X)), ("model", clf)])


def crossfit_propensity(
    X: pd.DataFrame,
    t: ArrayLike,
    model: BaseEstimator | None = None,
    n_folds: int = 5,
    seed: int = 0,
    clip: float = 1e-3,
) -> FloatArray:
    """Out-of-fold estimates of ``P(T = 1 | X)``, clipped to ``[clip, 1 - clip]``."""
    t = np.asarray(t, int)
    model = model if model is not None else propensity_model(X)
    scores = np.empty(len(t))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in folds.split(X, t):
        fitted = clone(model).fit(X.iloc[tr], t[tr])
        scores[te] = fitted.predict_proba(X.iloc[te])[:, 1]
    return np.clip(scores, clip, 1 - clip)
