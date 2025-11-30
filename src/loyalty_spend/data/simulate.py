"""Semi-synthetic benchmark calibrated to the training data.

The supplied training data is almost noise free: a linear model explains
more than 99.9% of the variance, so it cannot show when flexible models are
worth their cost. The simulator keeps the joint distribution of the real
features (rows are resampled) and the fitted linear structure, and adds
controlled departures from it: measurement noise, heteroscedasticity, smooth
nonlinear effects and interactions. Because the noise is known, every model
can be compared with the irreducible error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from loyalty_spend.data.schema import FEATURES, ID, TARGET
from loyalty_spend.features.preprocess import split_xy


@dataclass(frozen=True)
class Scenario:
    """Departures from the linear reference model.

    Parameters
    ----------
    name : str
    noise : float
        Noise standard deviation as a share of the target's standard deviation.
    heteroscedastic : bool
        Scale the noise with the reference prediction, so high spenders are
        less predictable.
    nonlinear : float
        Weight of the smooth nonlinear terms, in units of the target's
        standard deviation.
    interaction : float
        Weight of the interaction terms, in the same units.
    month_noise : bool
        Scale the noise of each joining month by the ratio of that month's
        residual SD to the pooled residual SD of the reference fit on the
        real data, keeping the mean noise variance, so the fivefold spread
        between months seen in ``train.csv`` carries over.
    """

    name: str
    noise: float = 0.05
    heteroscedastic: bool = False
    nonlinear: float = 0.0
    interaction: float = 0.0
    month_noise: bool = False


SCENARIOS: dict[str, Scenario] = {
    "linear": Scenario("linear", noise=0.05),
    "noisy": Scenario("noisy", noise=0.20, heteroscedastic=True),
    "nonlinear": Scenario("nonlinear", noise=0.05, nonlinear=0.25),
    "interactions": Scenario(
        "interactions", noise=0.05, nonlinear=0.25, interaction=0.25
    ),
    "month noise": Scenario("month noise", noise=0.05, month_noise=True),
}


def month_noise_ratios(X: pd.DataFrame, residuals: np.ndarray) -> dict[str, float]:
    """Residual SD of each joining month over the pooled residual SD."""
    months = X["joining_month"].astype(str).to_numpy()
    pooled = float(np.sqrt(np.mean(residuals**2)))
    return {
        m: float(np.sqrt(np.mean(residuals[months == m] ** 2)) / pooled)
        for m in np.unique(months)
    }


def _standardize(x: pd.Series) -> np.ndarray:
    x = x.astype(float).fillna(x.median())
    return ((x - x.mean()) / (x.std() or 1.0)).to_numpy()


def nonlinear_terms(X: pd.DataFrame) -> np.ndarray:
    """Saturating basket-size effect and a threshold in first-month spend."""
    items = _standardize(X["items_in_first_month"])
    first = _standardize(X["first_month"])
    return np.tanh(2.0 * items) + 1.5 * (first > 0.8) - 0.5 * first**2


def interaction_terms(X: pd.DataFrame) -> np.ndarray:
    """Promotion joiners in their first year, and tenure by first-month spend."""
    promo_new = (X["promotion"].astype(str) == "Yes") & (
        X["loyalty_years"].astype(str) == "0-1"
    )
    long_tenure = X["loyalty_years"].astype(str).isin(["5-10", "10+"])
    first = _standardize(X["first_month"])
    return 1.5 * promo_new.to_numpy() - 1.0 * long_tenure.to_numpy() * first


def _centred(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


@dataclass
class SimulatedData:
    frame: pd.DataFrame
    signal: np.ndarray
    noise_sd: np.ndarray

    @property
    def bayes_rmse(self) -> float:
        """Root mean noise variance; no model can do better in expectation."""
        return float(np.sqrt(np.mean(self.noise_sd**2)))


def simulate(
    train: pd.DataFrame,
    scenario: Scenario,
    n: int = 2000,
    seed: int = 0,
) -> SimulatedData:
    """Draw ``n`` customers under ``scenario``.

    Features are resampled rows of ``train``. The signal is the in-sample
    OLS fit on ``train`` plus the scenario's nonlinear and interaction terms;
    noise is Student-t with five degrees of freedom, rescaled to the
    requested standard deviation, to allow occasional large deviations.
    """
    from loyalty_spend.models.linear import ols

    rng = np.random.default_rng(seed)
    X_ref, y_ref = split_xy(train)
    reference = ols().fit(X_ref, y_ref)
    scale = float(y_ref.std())

    rows = rng.integers(0, len(train), size=n)
    X = X_ref.iloc[rows].reset_index(drop=True)
    signal = reference.predict(X)
    if scenario.nonlinear:
        signal = signal + scenario.nonlinear * scale * _centred(nonlinear_terms(X))
    if scenario.interaction:
        signal = signal + scenario.interaction * scale * _centred(interaction_terms(X))

    sd = np.full(n, scenario.noise * scale)
    if scenario.heteroscedastic:
        sd = sd * signal / signal.mean()
    if scenario.month_noise:
        ratios = month_noise_ratios(X_ref, y_ref.to_numpy() - reference.predict(X_ref))
        factor = X["joining_month"].astype(str).map(ratios).fillna(1.0).to_numpy()
        sd = sd * factor / np.sqrt(np.mean(factor**2))
    t = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    spend = np.clip(signal + sd * t, 0.01, None).round(2)

    frame = X.assign(**{TARGET: spend})
    frame.insert(0, ID, np.arange(1, n + 1))
    return SimulatedData(frame[[ID, TARGET, *FEATURES]], signal, sd)
