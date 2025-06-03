"""Mondrian (group-conditional) split conformal prediction.

Split conformal intervals have the right coverage on average over all
customers, but the README shows that they over-cover low spenders and
under-cover high spenders when the noise grows with spend. Mondrian conformal
prediction (Vovk et al., 2005) calibrates separately within groups, which
gives the finite-sample guarantee within every group:

    P(Y in C(X) | G(X) = g) >= 1 - alpha   for each group g.

Groups can be a feature (for example tenure band) or bins of the point
forecast, which targets exactly the decile pattern seen in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split

from loyalty_spend.uncertainty.conformal import Intervals, conformal_quantile

PREDICTION = "prediction"


class MondrianConformalRegressor:
    """Split conformal prediction with one quantile per group.

    Parameters
    ----------
    estimator : estimator
    by : str
        A feature column, or ``"prediction"`` for bins of the point forecast.
    n_bins : int
        Number of prediction bins (quantiles of the calibration forecasts).
    min_group_size : int
        Groups with fewer calibration points use the pooled quantile; with
        very few points the group quantile would be infinite.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        by: str = PREDICTION,
        alpha: float = 0.1,
        calibration_size: float = 0.25,
        n_bins: int = 5,
        min_group_size: int = 30,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.by = by
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.n_bins = n_bins
        self.min_group_size = min_group_size
        self.seed = seed

    def _groups(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        if self.by == PREDICTION:
            return np.searchsorted(self.edges_, pred, side="right")
        return X[self.by].astype(str).fillna("<missing>").to_numpy()

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> MondrianConformalRegressor:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.seed
        )
        self.model_ = clone(self.estimator).fit(X_fit, y_fit)
        pred = self.model_.predict(X_cal)
        if self.by == PREDICTION:
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.edges_ = np.unique(np.quantile(pred, qs))
        groups = self._groups(X_cal, pred)
        scores = np.abs(np.asarray(y_cal, float) - pred)
        self.pooled_q_ = conformal_quantile(scores, self.alpha)
        self.q_: dict[object, float] = {}
        self.group_sizes_: dict[object, int] = {}
        for g in np.unique(groups):
            s = scores[groups == g]
            self.group_sizes_[g] = len(s)
            if len(s) >= self.min_group_size:
                self.q_[g] = conformal_quantile(s, self.alpha)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def predict_interval(self, X: pd.DataFrame) -> Intervals:
        pred = self.predict(X)
        groups = self._groups(X, pred)
        half = np.array([self.q_.get(g, self.pooled_q_) for g in groups])
        return Intervals(pred - half, pred + half)
