"""Reward cost forecasting for the loyalty programme.

Rewards are a nonlinear function of spend: a customer just below a tier
threshold earns much less than one just above it. Applying the schedule to
point forecasts therefore misstates the expected cost; by Jensen's inequality
``E[r(S)]`` differs from ``r(E[S])`` wherever the schedule is not linear. The
forecast below propagates predictive uncertainty through the schedule by
simulation and reports the distribution of total programme cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    def rate(self, spend) -> np.ndarray:
        spend = np.asarray(spend, float)
        thresholds = np.array([t for t, _ in self.tiers])
        rates = np.array([r for _, r in self.tiers])
        idx = np.searchsorted(thresholds, spend, side="right") - 1
        return np.where(idx >= 0, rates[np.clip(idx, 0, None)], 0.0)

    def reward(self, spend) -> np.ndarray:
        spend = np.asarray(spend, float)
        return self.rate(spend) * np.clip(spend, 0.0, None)

    def tier(self, spend) -> np.ndarray:
        thresholds = np.array([t for t, _ in self.tiers])
        return np.searchsorted(thresholds, np.asarray(spend, float), side="right") - 1
