"""Semi-synthetic studies with a known promotion effect.

Observational estimators can only be checked where the truth is known.
Covariate rows are resampled from the training data, spend without the
promotion follows the OLS fit on those data, and uptake of the promotion
depends on the sign-up attributes through a logistic propensity. With
``confounding > 0`` members with a high expected spend are more likely to
take the promotion, so the naive difference in means overstates its effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from loyalty_spend.causal.data import PRE_TREATMENT, TREATMENT
from loyalty_spend.data.schema import FEATURES, ID, TARGET
from loyalty_spend.features.preprocess import split_xy


@dataclass
class SimulatedStudy:
    frame: pd.DataFrame
    effect: np.ndarray
    propensity: np.ndarray

    @property
    def true_ate(self) -> float:
        return float(self.effect.mean())

    @property
    def true_att(self) -> float:
        treated = self.frame[TREATMENT].to_numpy() == "Yes"
        return float(self.effect[treated].mean())


def _pre_treatment_signal(train: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"), LinearRegression()
    ).fit(train[PRE_TREATMENT].astype(str), train[TARGET].astype(float))
    signal = model.predict(X[PRE_TREATMENT].astype(str))
    sd = signal.std()
    return (signal - signal.mean()) / sd if sd > 0 else signal * 0.0


def simulate_promotion(
    train: pd.DataFrame,
    n: int = 2000,
    effect: float = 5.0,
    confounding: float = 1.0,
    treated_share: float = 0.4,
    noise: float = 0.1,
    seed: int = 0,
) -> SimulatedStudy:
    """Draw a study with a constant promotion effect.

    Parameters
    ----------
    train : DataFrame
        Cleaned training data; supplies the covariate distribution and the
        outcome model.
    effect : float
        Effect of the promotion on annual spend.
    confounding : float
        Log-odds of uptake per standard deviation of expected spend from the
        sign-up attributes.
    treated_share : float
        Share of members taking the promotion when ``confounding`` is zero.
    noise : float
        Noise standard deviation as a share of the spend standard deviation.
    """
    from loyalty_spend.models.linear import ols

    rng = np.random.default_rng(seed)
    X_ref, y_ref = split_xy(train)
    no_promo = X_ref.assign(**{TREATMENT: "No"})
    reference = ols().fit(X_ref, y_ref)
    rows = rng.integers(0, len(train), size=n)
    X = no_promo.iloc[rows].reset_index(drop=True)

    z = _pre_treatment_signal(train, X)
    propensity = expit(logit(treated_share) + confounding * z)
    t = rng.uniform(size=n) < propensity
    tau = np.full(n, float(effect))

    baseline = reference.predict(X)
    sd = noise * float(y_ref.std())
    spend = baseline + tau * t + rng.normal(scale=sd, size=n)

    frame = X.assign(**{TREATMENT: np.where(t, "Yes", "No")})
    frame[TARGET] = np.clip(spend, 0.01, None).round(2)
    frame.insert(0, ID, np.arange(1, n + 1))
    return SimulatedStudy(frame[[ID, TARGET, *FEATURES]], tau, propensity)
