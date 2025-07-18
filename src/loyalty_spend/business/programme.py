"""Reward programme design for a customer file in one call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from loyalty_spend.business.design import (
    behavioural_cost,
    compare_schedules,
    cost_sensitivity,
    search_thresholds,
)
from loyalty_spend.business.rewards import (
    MarginalSchedule,
    RewardSchedule,
    gaussian_sampler,
    simulate_liability,
)
from loyalty_spend.data.schema import ID
from loyalty_spend.features.preprocess import feature_frame, split_xy
from loyalty_spend.models.distributional import HeteroscedasticGaussianRegressor


@dataclass
class ProgrammeDesign:
    schedules: pd.DataFrame
    sensitivity: pd.DataFrame
    tail_risk: dict[str, float]
    behaviour: pd.DataFrame
    search: pd.DataFrame
    budget: float
    cost_to_spend: float


def design_programme(
    train: pd.DataFrame,
    customers: pd.DataFrame,
    tiers: tuple[tuple[float, float], ...],
    budget: float | None = None,
    offsets: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0),
    n_simulations: int = 2000,
    seed: int = 0,
) -> ProgrammeDesign:
    """Cost, risk and alternatives of a reward schedule for ``customers``.

    Spend is drawn from a heteroscedastic Gaussian model fitted to
    ``train``. Without a ``budget``, the 95th percentile of the current
    schedule's cost is used, so the search looks for schedules that reach
    more members at no more risk than today.
    """
    model = HeteroscedasticGaussianRegressor().fit(*split_xy(train))
    X = feature_frame(customers)
    mean, sigma = model.predict(X), model.predict_sigma(X)
    sampler = gaussian_sampler(mean, sigma)
    whole, marginal = RewardSchedule(tiers), MarginalSchedule(tiers)
    schedules = compare_schedules(
        sampler,
        {"whole spend": whole, "marginal": marginal},
        "whole spend",
        n_simulations,
        seed,
    )
    forecast = simulate_liability(
        customers[ID], mean, sampler, whole, n_simulations, seed
    )
    risk = forecast.tail_risk(0.95)
    budget = risk["var"] if budget is None else budget
    behaviour = pd.DataFrame(
        [
            {
                "schedule": name,
                **behavioural_cost(
                    sampler, sched, n_simulations=n_simulations, seed=seed
                ),
            }
            for name, sched in (("whole spend", whole), ("marginal", marginal))
        ]
    )
    return ProgrammeDesign(
        schedules=schedules,
        sensitivity=cost_sensitivity(
            sampler, whole, n_simulations=n_simulations, seed=seed
        ),
        tail_risk=risk,
        behaviour=behaviour,
        search=search_thresholds(
            sampler,
            whole,
            budget,
            np.asarray(offsets),
            n_simulations=n_simulations,
            seed=seed,
        ),
        budget=float(budget),
        cost_to_spend=forecast.cost_to_spend(),
    )
