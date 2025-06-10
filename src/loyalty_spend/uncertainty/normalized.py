r"""Normalized split conformal prediction.

The conformity score is the absolute residual divided by an estimate of how
hard the case is,

.. math::

    s_i = \frac{|y_i - \hat\mu(x_i)|}{\hat\sigma(x_i) + \beta},

so the interval :math:`\hat\mu(x) \pm \hat q\,(\hat\sigma(x) + \beta)`
widens where errors are large (Papadopoulos et al., 2008; Lei et al., 2018).
The difficulty model is fitted to out-of-fold absolute residuals of the
proper training set, because in-sample residuals of a flexible model are too
small and would make it underestimate the difficulty. The marginal coverage
guarantee holds for any difficulty model; a good one only makes the
intervals more adaptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor
from loyalty_spend.uncertainty.conformal import Intervals, conformal_quantile


def default_difficulty_model(seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor("ordinal")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    random_state=seed,
                ),
            ),
        ]
    )


class NormalizedConformalRegressor:
    """Split conformal intervals scaled by a model of the absolute residual.

    Parameters
    ----------
    estimator : estimator
        Point model.
    difficulty : estimator, optional
        Model of ``|y - mu(x)|``; gradient boosting with the absolute loss by
        default.
    beta : float
        Added to the difficulty so that cases with a near-zero estimate do
        not get zero-width intervals.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        difficulty: BaseEstimator | None = None,
        alpha: float = 0.1,
        calibration_size: float = 0.25,
        beta: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.difficulty = difficulty
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.beta = beta
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> NormalizedConformalRegressor:
        y = np.asarray(y, float)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.seed
        )
        self.model_ = clone(self.estimator).fit(X_fit, y_fit)
        oof = cross_val_predict(clone(self.estimator), X_fit, y_fit, cv=5)
        difficulty = self.difficulty or default_difficulty_model(self.seed)
        self.difficulty_ = clone(difficulty).fit(X_fit, np.abs(y_fit - oof))
        scores = np.abs(y_cal - self.model_.predict(X_cal)) / self._sigma(X_cal)
        self.q_ = conformal_quantile(scores, self.alpha)
        return self

    def _sigma(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.difficulty_.predict(X), 0.0, None) + self.beta

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def predict_interval(self, X: pd.DataFrame) -> Intervals:
        pred = self.predict(X)
        half = self.q_ * self._sigma(X)
        return Intervals(pred - half, pred + half)
