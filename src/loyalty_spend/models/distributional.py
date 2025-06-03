r"""Distributional regression: a full predictive distribution per customer.

The heteroscedastic Gaussian model assumes

.. math::

    y_i \mid x_i \sim \mathcal N\big(x_i^\top\beta,\; \exp(z_i^\top\gamma)\big),

fitted in two stages: least squares for the mean, then a ridge regression of
the log squared residuals for the variance. The log of a squared Gaussian
residual is biased downwards by :math:`\mathbb E[\log\chi^2_1] \approx -1.27`,
so the variance is rescaled until the standardized residuals have unit mean
square on the training data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor


def _linear(model: BaseEstimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor("onehot")), ("model", model)])


class HeteroscedasticGaussianRegressor(RegressorMixin, BaseEstimator):
    """Linear mean and log-linear variance.

    Parameters
    ----------
    mean_model, variance_model : estimator, optional
        Pipelines for the mean and for the log squared residuals. The
        defaults are OLS and ridge on the standard one-hot design.
    min_sigma : float
        Floor on the predicted standard deviation.
    """

    def __init__(
        self,
        mean_model: BaseEstimator | None = None,
        variance_model: BaseEstimator | None = None,
        min_sigma: float = 1e-3,
    ) -> None:
        self.mean_model = mean_model
        self.variance_model = variance_model
        self.min_sigma = min_sigma

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> HeteroscedasticGaussianRegressor:
        y = np.asarray(y, float)
        mean = self.mean_model or _linear(LinearRegression())
        var = self.variance_model or _linear(RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self.mean_ = clone(mean).fit(X, y)
        resid = y - self.mean_.predict(X)
        log_sq = np.log(np.maximum(resid**2, 1e-12))
        self.variance_ = clone(var).fit(X, log_sq)
        raw = np.exp(self.variance_.predict(X))
        self.scale_ = float(np.mean(resid**2 / raw))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.mean_.predict(X)

    def predict_sigma(self, X: pd.DataFrame) -> np.ndarray:
        var = self.scale_ * np.exp(self.variance_.predict(X))
        return np.maximum(np.sqrt(var), self.min_sigma)

    def predict_quantiles(self, X: pd.DataFrame, taus: Sequence[float]) -> np.ndarray:
        mu, sigma = self.predict(X), self.predict_sigma(X)
        z = stats.norm.ppf(np.asarray(taus, float))
        return mu[:, None] + sigma[:, None] * z[None, :]

    def sample(self, X: pd.DataFrame, n: int, seed: int = 0) -> np.ndarray:
        """``n`` draws per row, shape ``(len(X), n)``."""
        rng = np.random.default_rng(seed)
        mu, sigma = self.predict(X), self.predict_sigma(X)
        return mu[:, None] + sigma[:, None] * rng.standard_normal((len(mu), n))
