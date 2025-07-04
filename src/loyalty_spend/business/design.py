"""Comparing and designing reward schedules.

Two schedules are compared on the same simulated spend (common random
numbers), so the difference in their cost is not blurred by two independent
sets of draws. The variance of a paired difference is
``Var(A) + Var(B) - 2 Cov(A, B)``, and the covariance is large because both
schedules pay more when spend is high.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.business.rewards import MarginalSchedule, RewardSchedule, Sampler

Schedule = RewardSchedule | MarginalSchedule


def schedule_cost_draws(
    sampler: Sampler,
    schedules: Mapping[str, Schedule],
    n_simulations: int = 5000,
    seed: int = 0,
    batch: int = 500,
) -> dict[str, np.ndarray]:
    """Total cost of every schedule on shared draws of spend."""
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n_simulations) for name in schedules}
    for start in range(0, n_simulations, batch):
        m = min(batch, n_simulations - start)
        spend = np.clip(sampler(rng, m), 0.0, None)
        for name, schedule in schedules.items():
            out[name][start : start + m] = schedule.reward(spend).sum(axis=1)
    return out


def compare_schedules(
    sampler: Sampler,
    schedules: Mapping[str, Schedule],
    reference: str,
    n_simulations: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Expected cost of each schedule and its paired difference to ``reference``.

    ``difference_se`` is the Monte Carlo standard error of the expected
    difference, which with common random numbers is much smaller than the
    spread of either total.
    """
    if reference not in schedules:
        raise KeyError(f"reference '{reference}' not among the schedules")
    draws = schedule_cost_draws(sampler, schedules, n_simulations, seed)
    base = draws[reference]
    rows = []
    for name, d in draws.items():
        diff = d - base
        rows.append(
            {
                "schedule": name,
                "expected_cost": float(d.mean()),
                "cost_sd": float(d.std(ddof=1)),
                "p05": float(np.quantile(d, 0.05)),
                "p95": float(np.quantile(d, 0.95)),
                "difference": float(diff.mean()),
                "difference_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            }
        )
    return pd.DataFrame(rows)


def independent_difference_se(a: ArrayLike, b: ArrayLike) -> float:
    """Standard error of a difference of means from independent draws."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
