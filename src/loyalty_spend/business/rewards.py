"""Reward cost forecasting for the loyalty programme.

Rewards are a nonlinear function of spend: a customer just below a tier
threshold earns much less than one just above it. Applying the schedule to
point forecasts therefore misstates the expected cost; by Jensen's inequality
``E[r(S)]`` differs from ``r(E[S])`` wherever the schedule is not linear. The
forecast below propagates predictive uncertainty through the schedule by
simulation and reports the distribution of total programme cost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class RewardSchedule:
    """Tiered reward rates.

    ``tiers`` holds ``(threshold, rate)`` pairs. A customer whose annual
    spend reaches ``threshold`` earns ``rate`` times the whole spend; the
    highest tier reached applies.
    """

    tiers: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        thresholds = [t for t, _ in self.tiers]
        if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
            raise ValueError("tier thresholds must be strictly increasing")
        if any(not 0.0 <= r < 1.0 for _, r in self.tiers):
            raise ValueError("reward rates must be in [0, 1)")

    def rate(self, spend: ArrayLike) -> np.ndarray:
        spend = np.asarray(spend, float)
        thresholds = np.array([t for t, _ in self.tiers])
        rates = np.array([r for _, r in self.tiers])
        idx = np.searchsorted(thresholds, spend, side="right") - 1
        return np.where(idx >= 0, rates[np.clip(idx, 0, None)], 0.0)

    def reward(self, spend: ArrayLike) -> np.ndarray:
        spend = np.asarray(spend, float)
        return self.rate(spend) * np.clip(spend, 0.0, None)

    def tier(self, spend: ArrayLike) -> np.ndarray:
        thresholds = np.array([t for t, _ in self.tiers])
        return np.searchsorted(thresholds, np.asarray(spend, float), side="right") - 1


@dataclass
class LiabilityForecast:
    per_customer: pd.DataFrame
    total_draws: np.ndarray

    def summary(self, levels: Sequence[float] = (0.05, 0.5, 0.95)) -> dict[str, float]:
        q = np.quantile(self.total_draws, levels)
        out = {
            "expected_total": float(self.total_draws.mean()),
            "plug_in_total": float(self.per_customer["plug_in_reward"].sum()),
        }
        out.update(
            {f"p{int(round(100 * lv)):02d}": float(v) for lv, v in zip(levels, q)}
        )
        return out


Sampler = Callable[[np.random.Generator, int], np.ndarray]


def residual_sampler(predicted_spend: ArrayLike, residuals: ArrayLike) -> Sampler:
    """Spend as point forecast plus a resampled out-of-sample residual."""
    pred = np.asarray(predicted_spend, float)
    resid = np.asarray(residuals, float)

    def draw(rng: np.random.Generator, m: int) -> np.ndarray:
        return pred[None, :] + rng.choice(resid, size=(m, len(pred)))

    return draw


def gaussian_sampler(mean: ArrayLike, sigma: ArrayLike) -> Sampler:
    """Spend from a per-customer normal predictive distribution."""
    mu = np.asarray(mean, float)
    sd = np.broadcast_to(np.asarray(sigma, float), mu.shape)

    def draw(rng: np.random.Generator, m: int) -> np.ndarray:
        return mu[None, :] + sd[None, :] * rng.standard_normal((m, len(mu)))

    return draw


def simulate_liability(
    customer_ids: ArrayLike,
    predicted_spend: ArrayLike,
    sampler: Sampler,
    schedule: RewardSchedule,
    n_simulations: int = 5000,
    seed: int = 0,
    batch: int = 500,
) -> LiabilityForecast:
    """Monte Carlo distribution of reward cost for any predictive sampler.

    ``sampler(rng, m)`` returns ``m`` joint draws of annual spend, one column
    per customer. Draws are clipped at zero, the schedule is applied to each
    draw and the rewards are summed over customers.
    """
    rng = np.random.default_rng(seed)
    pred = np.asarray(predicted_spend, float)
    n = len(pred)
    totals = np.empty(n_simulations)
    reward_sum = np.zeros(n)
    upgrade = np.zeros(n)
    base_tier = schedule.tier(pred)
    for start in range(0, n_simulations, batch):
        m = min(batch, n_simulations - start)
        spend = np.clip(sampler(rng, m), 0.0, None)
        rewards = schedule.reward(spend)
        totals[start : start + m] = rewards.sum(axis=1)
        reward_sum += rewards.sum(axis=0)
        upgrade += (schedule.tier(spend) != base_tier[None, :]).sum(axis=0)
    per_customer = pd.DataFrame(
        {
            "customer_id": np.asarray(customer_ids),
            "predicted_spend": pred,
            "plug_in_reward": schedule.reward(pred),
            "expected_reward": reward_sum / n_simulations,
            "tier_change_probability": upgrade / n_simulations,
        }
    )
    return LiabilityForecast(per_customer, totals)


def forecast_liability(
    customer_ids: ArrayLike,
    predicted_spend: ArrayLike,
    residuals: ArrayLike,
    schedule: RewardSchedule,
    n_simulations: int = 5000,
    seed: int = 0,
    batch: int = 500,
) -> LiabilityForecast:
    """Monte Carlo distribution of reward cost with a residual bootstrap.

    Spend for each customer is simulated as ``prediction + e`` with ``e``
    drawn from out-of-sample residuals, which assumes the same error
    distribution for every customer. Use ``simulate_liability`` with
    ``gaussian_sampler`` or another sampler when the uncertainty differs
    between customers.
    """
    return simulate_liability(
        customer_ids,
        predicted_spend,
        residual_sampler(predicted_spend, residuals),
        schedule,
        n_simulations,
        seed,
        batch,
    )


def near_threshold(
    forecast: LiabilityForecast, min_probability: float = 0.2
) -> pd.DataFrame:
    """Customers whose tier is uncertain, the main source of cost variance."""
    pc = forecast.per_customer
    return pc[pc["tier_change_probability"] >= min_probability].sort_values(
        "tier_change_probability", ascending=False
    )
