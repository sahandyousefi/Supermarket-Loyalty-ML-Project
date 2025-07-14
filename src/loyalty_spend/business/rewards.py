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
    spend_draws: np.ndarray | None = None
    tier_shares: np.ndarray | None = None

    def standard_errors(
        self, levels: Sequence[float] = (0.05, 0.5, 0.95), n_batches: int = 20
    ) -> dict[str, float]:
        """Monte Carlo standard errors of the summary statistics.

        The mean uses the sample standard deviation over draws. Quantiles use
        batch means: the draws are cut into ``n_batches`` consecutive
        batches, the quantile is computed in each, and the spread of the
        batch quantiles divided by the square root of the number of batches
        estimates the error of the full-sample quantile.
        """
        draws = self.total_draws
        out = {"expected_total": float(draws.std(ddof=1) / np.sqrt(len(draws)))}
        batches = np.array_split(draws, n_batches)
        for lv in levels:
            per_batch = np.array([np.quantile(b, lv) for b in batches])
            out[f"p{int(round(100 * lv)):02d}"] = float(
                per_batch.std(ddof=1) / np.sqrt(n_batches)
            )
        return out

    def tail_risk(self, level: float = 0.95) -> dict[str, float]:
        """Value at risk and expected shortfall of total reward cost.

        ``var`` is the ``level`` quantile of the cost, the budget that is
        exceeded with probability ``1 - level``; ``expected_shortfall`` is
        the mean cost in the draws at or above it, the average overrun when
        that budget is exceeded (Acerbi and Tasche, 2002).
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        draws = self.total_draws
        var = float(np.quantile(draws, level))
        es = float(draws[draws >= var].mean())
        mean = float(draws.mean())
        return {
            "level": level,
            "var": var,
            "expected_shortfall": es,
            "var_over_expected": var - mean,
            "shortfall_over_expected": es - mean,
        }

    def cost_to_spend(self) -> float:
        """Expected reward cost as a share of expected total spend."""
        if self.spend_draws is None:
            raise ValueError("the forecast was made without spend totals")
        return float(self.total_draws.mean() / self.spend_draws.mean())

    def ratio_interval(self, level: float = 0.9) -> tuple[float, float]:
        """Central interval of the cost-to-spend ratio over the draws."""
        if self.spend_draws is None:
            raise ValueError("the forecast was made without spend totals")
        ratio = self.total_draws / self.spend_draws
        lo, hi = np.quantile(ratio, [(1 - level) / 2, (1 + level) / 2])
        return float(lo), float(hi)

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


def gaussian_qmc_sampler(mean: ArrayLike, sigma: ArrayLike, seed: int = 0) -> Sampler:
    """Normal draws from a scrambled Sobol sequence.

    Quasi-Monte Carlo points fill the unit cube more evenly than independent
    uniforms, so means of smooth functions converge faster than the
    ``1 / sqrt(B)`` of plain Monte Carlo (Owen, 2003). Reward schedules have
    jumps, which limits the gain, but the total over many customers is
    smooth enough for a clear improvement. The sampler ignores the generator
    passed by the simulation and keeps its own sequence.
    """
    from scipy.stats import norm, qmc

    mu = np.asarray(mean, float)
    sd = np.broadcast_to(np.asarray(sigma, float), mu.shape)
    engine = qmc.Sobol(d=len(mu), scramble=True, seed=seed)

    def draw(rng: np.random.Generator, m: int) -> np.ndarray:
        u = np.clip(engine.random(m), 1e-12, 1 - 1e-12)
        return mu[None, :] + sd[None, :] * norm.ppf(u)

    return draw


def simulate_liability(
    customer_ids: ArrayLike,
    predicted_spend: ArrayLike,
    sampler: Sampler,
    schedule: RewardSchedule | MarginalSchedule,
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
    spend_totals = np.empty(n_simulations)
    reward_sum = np.zeros(n)
    upgrade = np.zeros(n)
    n_tiers = len(schedule.tiers)
    tier_counts = np.zeros(n_tiers + 1)
    base_tier = schedule.tier(pred)
    for start in range(0, n_simulations, batch):
        m = min(batch, n_simulations - start)
        spend = np.clip(sampler(rng, m), 0.0, None)
        rewards = schedule.reward(spend)
        tiers = schedule.tier(spend)
        totals[start : start + m] = rewards.sum(axis=1)
        spend_totals[start : start + m] = spend.sum(axis=1)
        reward_sum += rewards.sum(axis=0)
        upgrade += (tiers != base_tier[None, :]).sum(axis=0)
        # Index 0 counts spend below the first threshold (tier -1).
        tier_counts += np.bincount((tiers + 1).ravel(), minlength=n_tiers + 1)
    per_customer = pd.DataFrame(
        {
            "customer_id": np.asarray(customer_ids),
            "predicted_spend": pred,
            "plug_in_reward": schedule.reward(pred),
            "expected_reward": reward_sum / n_simulations,
            "tier_change_probability": upgrade / n_simulations,
        }
    )
    shares = tier_counts / tier_counts.sum()
    return LiabilityForecast(per_customer, totals, spend_totals, shares)


def forecast_liability(
    customer_ids: ArrayLike,
    predicted_spend: ArrayLike,
    residuals: ArrayLike,
    schedule: RewardSchedule | MarginalSchedule,
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


@dataclass(frozen=True)
class MarginalSchedule:
    """Tier rates that apply only to the spend inside each band.

    Like income tax brackets: with thresholds 100 and 125 and rates 2% and
    3%, a member who spends 130 earns 2% of 25 plus 3% of 5. The reward is
    continuous in spend, so members gain nothing by bunching just above a
    threshold and the cost has no jumps, which a whole-spend schedule does.
    """

    tiers: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        RewardSchedule(self.tiers)

    def tier(self, spend: ArrayLike) -> np.ndarray:
        return RewardSchedule(self.tiers).tier(spend)

    def reward(self, spend: ArrayLike) -> np.ndarray:
        s = np.clip(np.asarray(spend, float), 0.0, None)
        thresholds = [t for t, _ in self.tiers]
        rates = [r for _, r in self.tiers]
        upper = [*thresholds[1:], np.inf]
        total = np.zeros_like(s)
        for lo, hi, rate in zip(thresholds, upper, rates):
            total += rate * np.clip(np.minimum(s, hi) - lo, 0.0, None)
        return total

    def rate(self, spend: ArrayLike) -> np.ndarray:
        """Average rate, the reward as a share of spend."""
        s = np.asarray(spend, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, self.reward(s) / s, 0.0)
