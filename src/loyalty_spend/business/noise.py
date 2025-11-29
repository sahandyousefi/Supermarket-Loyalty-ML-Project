"""Reward cost of a batch under different models of forecast noise.

The same point forecasts can carry different cost distributions: the
residual bootstrap gives every member one error distribution, a Gaussian
model with a fitted variance function gives each member its own normal
spread, and the scaled residual bootstrap gives each member the error
scale of their joining month with the shape of the pooled errors. Tier
assignment, and therefore cost, is sensitive to the spread of members
near a threshold, so the choice matters most for the tail of the cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loyalty_spend.arrays import FloatArray
from loyalty_spend.business.rewards import (
    LiabilityForecast,
    RewardSchedule,
    Sampler,
    residual_sampler,
    scaled_residual_sampler,
    simulate_liability,
)
from loyalty_spend.data.schema import ID
from loyalty_spend.evaluation.cv import cross_validate, make_splits
from loyalty_spend.features.preprocess import feature_frame, split_xy
from loyalty_spend.models.registry import build_model
from loyalty_spend.models.weighted import (
    GroupWeightedRegression,
    standardized_residuals,
)


def noise_samplers(
    train: pd.DataFrame, batch: pd.DataFrame, seed: int = 0
) -> dict[str, tuple[FloatArray, Sampler]]:
    """Point forecasts and spend samplers of each noise model for ``batch``."""
    from loyalty_spend.business.rewards import gaussian_sampler
    from loyalty_spend.models.distributional import HeteroscedasticGaussianRegressor

    X, y = split_xy(train)
    Xb = feature_frame(batch)
    splits = make_splits(len(y), 5, 1, seed)
    oof = cross_validate("ols", build_model("ols"), X, y, splits).oof
    ols = build_model("ols").fit(X, y)
    mean_ols = np.asarray(ols.predict(Xb), float)
    het = HeteroscedasticGaussianRegressor().fit(X, y)
    mean_het, sd_het = het.predict(Xb), het.predict_sigma(Xb)
    wls = GroupWeightedRegression().fit(X, y)
    mean_wls, sd_wls = wls.predict(Xb), wls.predict_sd(Xb)
    pool = standardized_residuals(wls, X, y)
    return {
        "residual bootstrap": (mean_ols, residual_sampler(mean_ols, y - oof)),
        "Gaussian, fitted variance": (mean_het, gaussian_sampler(mean_het, sd_het)),
        "scaled residuals by month": (
            mean_wls,
            scaled_residual_sampler(mean_wls, sd_wls, pool),
        ),
    }


def compare_noise_models(
    train: pd.DataFrame,
    batch: pd.DataFrame,
    schedule: RewardSchedule,
    n_simulations: int = 5000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, LiabilityForecast]]:
    forecasts = {}
    rows = []
    for name, (mean, sampler) in noise_samplers(train, batch, seed).items():
        fc = simulate_liability(batch[ID], mean, sampler, schedule, n_simulations, seed)
        forecasts[name] = fc
        s = fc.summary()
        risk = fc.tail_risk(0.95)
        rows.append(
            {
                "noise model": name,
                "expected_total": s["expected_total"],
                "p05": s["p05"],
                "p95": s["p95"],
                "expected_shortfall_95": risk["expected_shortfall"],
                "members_near_threshold": int(
                    (fc.per_customer["tier_change_probability"] >= 0.2).sum()
                ),
            }
        )
    return pd.DataFrame(rows), forecasts
