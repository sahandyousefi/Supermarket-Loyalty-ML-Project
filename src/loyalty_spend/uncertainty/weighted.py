r"""Conformal prediction under covariate shift.

When the customers to be scored differ from the training customers in their
features but not in how spend depends on the features, exchangeability
fails and split conformal intervals lose their coverage. Tibshirani et al.
(2019) restore it by weighting the calibration scores with the likelihood
ratio :math:`w(x) = dP_{\text{new}}(x) / dP_{\text{train}}(x)`. For a new
point :math:`x`, the interval uses the :math:`1-\alpha` quantile of

.. math::

    \sum_{i=1}^n p_i(x)\,\delta_{s_i} + p_{n+1}(x)\,\delta_{+\infty},
    \qquad p_i(x) = \frac{w(X_i)}{\sum_{j} w(X_j) + w(x)},

which reduces to the usual split conformal quantile when all weights are
equal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split

from loyalty_spend.uncertainty.conformal import Intervals


def weighted_conformal_quantile(
    scores: ArrayLike,
    calibration_weights: ArrayLike,
    test_weights: ArrayLike,
    alpha: float,
) -> np.ndarray:
    """Weighted ``1 - alpha`` quantile for each test weight.

    Returns infinity for a test point whose own weight is so large that the
    calibration mass cannot reach ``1 - alpha``.
    """
    s = np.asarray(scores, float)
    w = np.asarray(calibration_weights, float)
    w_test = np.atleast_1d(np.asarray(test_weights, float))
    if s.shape != w.shape:
        raise ValueError("need one weight per calibration score")
    if np.any(w < 0) or np.any(w_test < 0):
        raise ValueError("weights must be non-negative")
    order = np.argsort(s)
    s, w = s[order], w[order]
    cum = np.cumsum(w)
    total = cum[-1] + w_test
    # First sorted score whose normalised cumulative mass reaches 1 - alpha.
    target = (1 - alpha) * total[:, None]
    idx = np.argmax(cum[None, :] >= target - 1e-12, axis=1)
    reached = cum[-1] >= (1 - alpha) * total - 1e-12
    return np.where(reached, s[idx], np.inf)


class WeightedConformalRegressor:
    """Split conformal intervals reweighted towards a new population.

    ``predict_interval`` estimates the likelihood ratio between the new rows
    and the calibration rows with the domain classifier used for drift
    monitoring, so the intervals adapt to the batch being scored.

    Parameters
    ----------
    estimator : estimator
    clip : float
        Weights are clipped to ``[1 / clip, clip]``; unclipped ratios from a
        well-separating classifier make the intervals infinite.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        alpha: float = 0.1,
        calibration_size: float = 0.25,
        clip: float = 20.0,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.clip = clip
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> WeightedConformalRegressor:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.seed
        )
        self.model_ = clone(self.estimator).fit(X_fit, y_fit)
        self.X_cal_ = X_cal.reset_index(drop=True)
        self.scores_ = np.abs(np.asarray(y_cal, float) - self.model_.predict(X_cal))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def weights(self, X_new: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Estimated likelihood ratios at the calibration rows and at ``X_new``."""
        from loyalty_spend.monitoring.domain import domain_classifier_test

        result = domain_classifier_test(
            self.X_cal_, X_new.reset_index(drop=True), seed=self.seed
        )
        ratio = result.density_ratio(clip=self.clip)
        n_cal = len(self.X_cal_)
        return ratio[:n_cal], ratio[n_cal:]

    def predict_interval(self, X_new: pd.DataFrame) -> Intervals:
        w_cal, w_new = self.weights(X_new)
        q = weighted_conformal_quantile(self.scores_, w_cal, w_new, self.alpha)
        pred = self.predict(X_new)
        return Intervals(pred - q, pred + q)
