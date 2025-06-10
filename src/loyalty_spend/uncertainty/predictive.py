r"""Split conformal predictive distributions.

A conformal predictive system turns any point model into a full predictive
distribution that is calibrated in probability under exchangeability (Vovk
et al., 2019). With signed calibration residuals
:math:`r_1 \le \dots \le r_n`, the split version gives for a new customer
with point forecast :math:`\hat y` the distribution of :math:`\hat y + R`,
where :math:`R` puts mass :math:`1/(n+1)` on each residual and the remaining
:math:`1/(n+1)` beyond the largest one. Its quantile at level :math:`\tau`
is therefore :math:`\hat y + r_{(\lceil \tau (n+1) \rceil)}`, and the PIT
of a new outcome is uniform up to a discreteness of :math:`1/(n+1)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split


def residual_quantiles(residuals: ArrayLike, taus: Sequence[float]) -> np.ndarray:
    """Conformal quantiles of sorted residuals, clipped to the observed range."""
    r = np.sort(np.asarray(residuals, float))
    n = len(r)
    k = np.ceil(np.asarray(taus, float) * (n + 1)).astype(int)
    return r[np.clip(k, 1, n) - 1]


class ConformalPredictiveDistribution:
    """Predictive distribution from a point model and held-out residuals."""

    def __init__(
        self,
        estimator: BaseEstimator,
        calibration_size: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.calibration_size = calibration_size
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> ConformalPredictiveDistribution:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.seed
        )
        self.model_ = clone(self.estimator).fit(X_fit, y_fit)
        self.residuals_ = np.sort(np.asarray(y_cal, float) - self.model_.predict(X_cal))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def quantiles(self, X: pd.DataFrame, taus: Sequence[float]) -> np.ndarray:
        shift = residual_quantiles(self.residuals_, taus)
        return self.predict(X)[:, None] + shift[None, :]

    def cdf(self, X: pd.DataFrame, y: ArrayLike) -> np.ndarray:
        """Predictive CDF at ``y`` for each row, using the upper conformal value."""
        r = self.residuals_
        z = np.asarray(y, float) - self.predict(X)
        return np.searchsorted(r, z, side="right") / (len(r) + 1)

    def sample(self, X: pd.DataFrame, n: int, seed: int = 0) -> np.ndarray:
        """Draw from the calibration residuals around each point forecast."""
        rng = np.random.default_rng(seed)
        draws = rng.choice(self.residuals_, size=(len(X), n))
        return self.predict(X)[:, None] + draws
