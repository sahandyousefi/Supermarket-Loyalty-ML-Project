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
    """

    name: str
    noise: float = 0.05
    heteroscedastic: bool = False
    nonlinear: float = 0.0
    interaction: float = 0.0


SCENARIOS: dict[str, Scenario] = {
    "linear": Scenario("linear", noise=0.05),
    "noisy": Scenario("noisy", noise=0.20, heteroscedastic=True),
    "nonlinear": Scenario("nonlinear", noise=0.05, nonlinear=0.25),
    "interactions": Scenario(
        "interactions", noise=0.05, nonlinear=0.25, interaction=0.25
    ),
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
