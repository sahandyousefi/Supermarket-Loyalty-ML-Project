"""Models that forecast several quantiles of spend at once.

Each level is fitted as its own model with the pinball loss of that level.
``predict`` returns the median so that the estimators also work wherever a
point forecast is expected; ``predict_quantiles`` returns one column per
level.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor

DEFAULT_TAUS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _check_taus(taus: Sequence[float]) -> tuple[float, ...]:
    t = tuple(float(v) for v in taus)
    if not t or any(not 0.0 < v < 1.0 for v in t) or list(t) != sorted(set(t)):
        raise ValueError("taus must be strictly increasing values in (0, 1)")
    return t


def linear_quantile_model(tau: float, alpha: float = 0.0) -> Pipeline:
    """Linear quantile regression solved exactly as a linear programme."""
    return Pipeline(
        [
            ("prep", make_preprocessor("onehot")),
            ("model", QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")),
        ]
    )


class MultiQuantileRegressor(RegressorMixin, BaseEstimator):
    """One model per quantile level.

    Parameters
    ----------
    factory : callable
        Maps a level ``tau`` to an unfitted estimator for that quantile.
    taus : sequence of float
        Strictly increasing levels in (0, 1); 0.5 must be among them so that
        ``predict`` can return the median.
    """

    def __init__(
        self,
        factory: Callable[[float], BaseEstimator] = linear_quantile_model,
        taus: Sequence[float] = DEFAULT_TAUS,
    ) -> None:
        self.factory = factory
        self.taus = taus

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> MultiQuantileRegressor:
        taus = _check_taus(self.taus)
        if 0.5 not in taus:
            raise ValueError("taus must include 0.5")
        self.taus_ = taus
        self.models_ = [self.factory(t).fit(X, y) for t in taus]
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.models_[self.taus_.index(0.5)].predict(X)
