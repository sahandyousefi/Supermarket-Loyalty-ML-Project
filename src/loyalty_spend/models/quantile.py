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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor

DEFAULT_TAUS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _check_taus(taus: Sequence[float]) -> tuple[float, ...]:
    t = tuple(float(v) for v in taus)
    if not t or any(not 0.0 < v < 1.0 for v in t) or list(t) != sorted(set(t)):
        raise ValueError("taus must be strictly increasing values in (0, 1)")
    return t


def crossing_rate(q: ArrayLike) -> float:
    """Share of rows whose quantile forecasts are not non-decreasing."""
    q = np.asarray(q, float)
    return float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))


def rearrange(q: ArrayLike) -> np.ndarray:
    """Monotone rearrangement of quantile forecasts.

    Separately fitted quantile models can cross, which makes the implied
    distribution invalid. Sorting each row is the rearrangement of
    Chernozhukov, Fernandez-Val and Galichon (2010): it restores monotonicity
    and never increases the distance to the true quantile function.
    """
    return np.sort(np.asarray(q, float), axis=1)


def linear_quantile_model(tau: float, alpha: float = 0.0) -> Pipeline:
    """Linear quantile regression solved exactly as a linear programme."""
    return Pipeline(
        [
            ("prep", make_preprocessor("onehot")),
            ("model", QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")),
        ]
    )


def gbm_quantile_model(tau: float, seed: int = 0) -> Pipeline:
    """Histogram gradient boosting with the pinball loss of level ``tau``."""
    prep = make_preprocessor("ordinal").set_output(transform="pandas")
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=tau,
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=seed,
    )
    return Pipeline([("prep", prep), ("model", model)])


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
        monotone: bool = True,
    ) -> None:
        self.factory = factory
        self.taus = taus
        self.monotone = monotone

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> MultiQuantileRegressor:
        taus = _check_taus(self.taus)
        if 0.5 not in taus:
            raise ValueError("taus must include 0.5")
        self.taus_ = taus
        self.models_ = [self.factory(t).fit(X, y) for t in taus]
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        q = np.column_stack([m.predict(X) for m in self.models_])
        return rearrange(q) if self.monotone else q

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(X)[:, self.taus_.index(0.5)]
