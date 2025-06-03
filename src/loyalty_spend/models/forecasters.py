"""Distributional forecasters behind one interface.

Every forecaster is fitted on a feature frame and returns quantiles on a
fixed grid of levels, so that CRPS, interval scores and PIT values can be
computed the same way for Gaussian, quantile and conformal methods.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from loyalty_spend.models.distributional import HeteroscedasticGaussianRegressor
from loyalty_spend.models.linear import ols
from loyalty_spend.models.quantile import (
    MultiQuantileRegressor,
    gbm_quantile_model,
    linear_quantile_model,
)

# Nineteen levels from 0.05 to 0.95; with the end points 0.05 and 0.95 the
# grid also gives the central 90% interval.
TAU_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))


class QuantileForecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: ArrayLike) -> QuantileForecaster: ...

    def quantiles(self, X: pd.DataFrame) -> np.ndarray: ...


class GaussianResidualForecaster:
    """OLS with a single residual standard deviation for everyone."""

    def __init__(self, taus: Sequence[float] = TAU_GRID) -> None:
        self.taus = tuple(taus)

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> GaussianResidualForecaster:
        y = np.asarray(y, float)
        self.model_ = ols().fit(X, y)
        resid = y - self.model_.predict(X)
        self.sigma_ = float(np.sqrt(np.sum(resid**2) / max(len(y) - 1, 1)))
        return self

    def quantiles(self, X: pd.DataFrame) -> np.ndarray:
        z = stats.norm.ppf(self.taus)
        return self.model_.predict(X)[:, None] + self.sigma_ * z[None, :]


class HeteroscedasticForecaster:
    def __init__(self, taus: Sequence[float] = TAU_GRID) -> None:
        self.taus = tuple(taus)

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> HeteroscedasticForecaster:
        self.model_ = HeteroscedasticGaussianRegressor().fit(X, y)
        return self

    def quantiles(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_quantiles(X, self.taus)


class QuantileModelForecaster:
    def __init__(
        self,
        factory: Callable[[float], object],
        taus: Sequence[float] = TAU_GRID,
    ) -> None:
        self.factory = factory
        self.taus = tuple(taus)

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> QuantileModelForecaster:
        self.model_ = MultiQuantileRegressor(self.factory, self.taus).fit(X, y)
        return self

    def quantiles(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_quantiles(X)


FORECASTERS: dict[str, Callable[[], QuantileForecaster]] = {
    "gaussian": GaussianResidualForecaster,
    "gaussian_heteroscedastic": HeteroscedasticForecaster,
    "linear_quantile": lambda: QuantileModelForecaster(linear_quantile_model),
    "gbm_quantile": lambda: QuantileModelForecaster(gbm_quantile_model),
}


def build_forecaster(name: str) -> QuantileForecaster:
    if name not in FORECASTERS:
        raise KeyError(f"unknown forecaster '{name}'; available: {sorted(FORECASTERS)}")
    return FORECASTERS[name]()
