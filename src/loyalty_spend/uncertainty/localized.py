r"""Localised conformal intervals with kernel weights.

Split conformal intervals use one quantile of the calibration scores for
every member, so members whose spend is harder to predict are under-covered
and the others over-covered. Localised conformal prediction (Guan, 2023)
weights each calibration score by its similarity to the member being
scored, :math:`w_i(x) = \exp(-d(X_i, x)/h)` with :math:`d` the Gower
distance, and takes the weighted quantile of the scores with the member's
own weight :math:`w(x, x) = 1` placed at infinity, as in weighted conformal
prediction. A small bandwidth :math:`h` calibrates on near neighbours and
moves towards conditional coverage; a large one recovers split conformal.

Without the level correction of Guan (2023) or the randomisation of Hore
and Barber (2023), the finite-sample marginal guarantee is lost, so the
marginal coverage is checked empirically together with the conditional
coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split

from loyalty_spend.synthetic.privacy import gower_cross
from loyalty_spend.uncertainty.conformal import Intervals
from loyalty_spend.uncertainty.weighted import weighted_conformal_quantile


class LocalizedConformalRegressor:
    def __init__(
        self,
        estimator: BaseEstimator,
        alpha: float = 0.1,
        bandwidth: float = 0.05,
        calibration_size: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.calibration_size = calibration_size
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> LocalizedConformalRegressor:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.seed
        )
        self.model_ = clone(self.estimator).fit(X_fit, y_fit)
        self.X_cal_ = X_cal.reset_index(drop=True)
        self.scores_ = np.abs(np.asarray(y_cal, float) - self.model_.predict(X_cal))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def weights(self, X_new: pd.DataFrame) -> np.ndarray:
        """Kernel weights, one row per new member and one column per score."""
        d = gower_cross(X_new.reset_index(drop=True), self.X_cal_)
        return np.exp(-np.nan_to_num(d, nan=1.0) / self.bandwidth)

    def effective_size(self, X_new: pd.DataFrame) -> np.ndarray:
        """Kish effective number of calibration scores behind each interval."""
        w = self.weights(X_new)
        return w.sum(axis=1) ** 2 / (w**2).sum(axis=1)

    def predict_interval(self, X_new: pd.DataFrame) -> Intervals:
        W = self.weights(X_new)
        q = np.array(
            [
                weighted_conformal_quantile(self.scores_, w, [1.0], self.alpha)[0]
                for w in W
            ]
        )
        pred = self.predict(X_new)
        return Intervals(pred - q, pred + q)
