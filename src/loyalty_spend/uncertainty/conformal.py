"""Distribution-free prediction intervals.

A point forecast of spend is not enough to budget for rewards; the finance
team needs a range with a stated probability of containing the outcome.
Conformal methods give intervals with finite-sample marginal coverage of at
least ``1 - alpha`` under exchangeability, for any underlying model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of the scores.

    Returns infinity when the calibration set is too small for the level.
    """
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(scores[k - 1])


@dataclass
class Intervals:
    lower: np.ndarray
    upper: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def coverage(self, y: ArrayLike) -> float:
        y = np.asarray(y, float)
        return float(np.mean((y >= self.lower) & (y <= self.upper)))


class SplitConformalRegressor:
    """Split conformal prediction with absolute residual scores.

    The model is fitted on one part of the data and the residuals on the
    other part set a constant half-width (Lei et al., 2018).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        alpha: float = 0.1,
        calibration_size: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> SplitConformalRegressor:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.seed
        )
        self.model_ = clone(self.estimator).fit(X_fit, y_fit)
        scores = np.abs(np.asarray(y_cal, float) - self.model_.predict(X_cal))
        self.q_ = conformal_quantile(scores, self.alpha)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def predict_interval(self, X: pd.DataFrame) -> Intervals:
        pred = self.predict(X)
        return Intervals(pred - self.q_, pred + self.q_)


def cv_plus_intervals(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    X_new: pd.DataFrame,
    alpha: float = 0.1,
    n_splits: int = 10,
    seed: int = 0,
) -> Intervals:
    """CV+ intervals (Barber et al., 2021).

    Uses every row for both fitting and calibration. With ``K`` folds the
    guaranteed coverage is ``1 - 2 alpha - sqrt(2/n)``; in practice it is
    close to ``1 - alpha``.
    """
    y = np.asarray(y, float)
    n = len(y)
    lower = np.empty((n, len(X_new)))
    upper = np.empty((n, len(X_new)))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        model = clone(estimator).fit(X.iloc[tr], y[tr])
        resid = np.abs(y[te] - model.predict(X.iloc[te]))
        pred_new = model.predict(X_new)
        lower[te] = pred_new[None, :] - resid[:, None]
        upper[te] = pred_new[None, :] + resid[:, None]
    k_lo = int(np.floor(alpha * (n + 1)))
    k_hi = int(np.ceil((1 - alpha) * (n + 1)))
    lo = np.sort(lower, axis=0)[max(k_lo - 1, 0)]
    hi = np.sort(upper, axis=0)[min(k_hi - 1, n - 1)]
    return Intervals(lo, hi)


def _quantile_model(q: float, seed: int) -> Pipeline:
    prep = make_preprocessor("ordinal").set_output(transform="pandas")
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=seed,
    )
    return Pipeline([("prep", prep), ("model", model)])


class ConformalizedQuantileRegressor:
    """Conformalized quantile regression (Romano et al., 2019).

    Gradient-boosted quantile models estimate the ``alpha/2`` and
    ``1 - alpha/2`` quantiles, and a conformal correction on held-out data
    widens or narrows the band so that coverage is guaranteed. Unlike split
    conformal with absolute residuals, the width adapts to each customer.
    """

    def __init__(
        self, alpha: float = 0.1, calibration_size: float = 0.25, seed: int = 0
    ) -> None:
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> ConformalizedQuantileRegressor:
        y = np.asarray(y, float)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.seed
        )
        self.lower_ = _quantile_model(self.alpha / 2, self.seed).fit(X_fit, y_fit)
        self.upper_ = _quantile_model(1 - self.alpha / 2, self.seed).fit(X_fit, y_fit)
        lo, hi = self.lower_.predict(X_cal), self.upper_.predict(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        self.q_ = conformal_quantile(scores, self.alpha)
        return self

    def predict_interval(self, X: pd.DataFrame) -> Intervals:
        lo, hi = self.lower_.predict(X), self.upper_.predict(X)
        return Intervals(np.minimum(lo, hi) - self.q_, np.maximum(lo, hi) + self.q_)


def coverage_by_group(
    intervals: Intervals, y: ArrayLike, groups: pd.Series
) -> pd.DataFrame:
    """Coverage and mean width per group, to check conditional coverage."""
    df = pd.DataFrame(
        {
            "group": np.asarray(groups),
            "covered": (np.asarray(y) >= intervals.lower)
            & (np.asarray(y) <= intervals.upper),
            "width": intervals.width,
        }
    )
    out = df.groupby("group", observed=True).agg(
        n=("covered", "size"),
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
    )
    return out.reset_index()
