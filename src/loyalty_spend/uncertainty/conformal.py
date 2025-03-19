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
from sklearn.base import clone
from sklearn.model_selection import train_test_split


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

    def coverage(self, y) -> float:
        y = np.asarray(y, float)
        return float(np.mean((y >= self.lower) & (y <= self.upper)))


class SplitConformalRegressor:
    """Split conformal prediction with absolute residual scores.

    The model is fitted on one part of the data and the residuals on the
    other part set a constant half-width (Lei et al., 2018).
    """

    def __init__(
        self,
        estimator,
        alpha: float = 0.1,
        calibration_size: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.seed = seed

    def fit(self, X: pd.DataFrame, y) -> SplitConformalRegressor:
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
