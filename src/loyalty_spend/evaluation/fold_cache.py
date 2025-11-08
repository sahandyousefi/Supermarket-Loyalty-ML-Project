"""Preprocessing fitted once per fold and shared between fits.

Hyperparameter search refits the whole pipeline for every trial on the same
folds, and a model comparison refits the same preprocessing for every model
that uses it. Only the final estimator changes, so the preprocessing of each
fold is fitted once, keyed by its parameters, and the transformed training
and test data are reused. A pipeline whose first step is not named
``prep`` is fitted as a whole, unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from loyalty_spend.arrays import FloatArray, Split


def split_pipeline(estimator: BaseEstimator) -> tuple[Any, Any] | None:
    """The ``prep`` step and the rest of a two-part pipeline, if it is one."""
    if not isinstance(estimator, Pipeline) or estimator.steps[0][0] != "prep":
        return None
    rest = estimator.steps[1:]
    final = rest[0][1] if len(rest) == 1 else Pipeline(rest)
    return estimator.steps[0][1], final


class FoldCache:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[Split],
    ) -> None:
        self.X, self.y, self.splits = X, y, splits
        self._store: dict[tuple[str, int], tuple[Any, Any, Any]] = {}
        self.hits = 0
        self.misses = 0

    def transformed(self, prep: BaseEstimator, fold: int) -> tuple[Any, Any, Any]:
        """Fitted ``prep`` and the transformed training and test features."""
        key = (repr(sorted(prep.get_params(deep=True).items(), key=str)), fold)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        tr, te = self.splits[fold]
        fitted = clone(prep).fit(self.X.iloc[tr], self.y.iloc[tr])
        value = (
            fitted,
            fitted.transform(self.X.iloc[tr]),
            fitted.transform(self.X.iloc[te]),
        )
        self._store[key] = value
        return value

    def fit_predict(
        self, estimator: BaseEstimator, fold: int
    ) -> tuple[BaseEstimator, FloatArray]:
        """Fitted estimator (a pipeline again) and its test-fold predictions."""
        tr, te = self.splits[fold]
        parts = split_pipeline(estimator)
        if parts is None:
            model = clone(estimator).fit(self.X.iloc[tr], self.y.iloc[tr])
            return model, np.asarray(model.predict(self.X.iloc[te]), float)
        prep, final = parts
        fitted_prep, Xt_tr, Xt_te = self.transformed(prep, fold)
        fitted_final = clone(final).fit(Xt_tr, self.y.iloc[tr])
        pred = np.asarray(fitted_final.predict(Xt_te), float)
        # The fitted preprocessing is shared between pipelines; do not refit it.
        return Pipeline([("prep", fitted_prep), ("model", fitted_final)]), pred
