"""Operating characteristics of the experiment by simulation on members.

Members are drawn with replacement from the observed file and enrolled one
after another. Assignment is stratified by tenure band and region within
blocks of enrolment, a known effect is added to the spend of treated
members, and the trial is analysed at every look with the chosen
estimator and stopped at the first crossing of the boundary. Repeating
this gives the type I error and power of the whole procedure, the expected
number of members, and the bias of the estimate at stopping, which a
sequential design introduces because large early estimates are the ones
that stop a trial.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from loyalty_spend.experiment.adjustment import (
    EffectEstimate,
    difference_in_means,
    lin_adjusted,
)
from loyalty_spend.experiment.randomisation import stratified_assignment

COVARIATES = [
    "first_month",
    "items_in_first_month",
    "region",
    "loyalty_years",
    "joining_month",
]


def _estimate(
    frame: pd.DataFrame, y: np.ndarray, t: np.ndarray, method: str
) -> EffectEstimate:
    if method == "difference":
        return difference_in_means(y, t)
    X = frame[COVARIATES].copy()
    for col in X.select_dtypes("category"):
        X[col] = X[col].astype(str)
    return lin_adjusted(y, X, t)


def simulate_trial(
    data: pd.DataFrame,
    n_total: int,
    effect: float,
    fractions: Sequence[float],
    bounds: Sequence[float],
    method: str = "lin",
    n_reps: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    looks = [int(round(f * n_total)) for f in fractions]
    rows = []
    for r in range(n_reps):
        sample = data.iloc[rng.integers(0, len(data), n_total)].reset_index(drop=True)
        t = np.zeros(n_total, bool)
        start = 0
        for end in looks:
            # Each enrolment block is randomised within strata on its own.
            block = sample.iloc[start:end][["loyalty_years", "region"]]
            t[start:end] = stratified_assignment(block, 0.5, seed=seed * 7919 + r)
            start = end
        y = sample["spend"].to_numpy(float) + effect * t
        row: dict[str, float] = {"rep": r, "stopped_at": len(looks), "rejected": False}
        for k, (n, c) in enumerate(zip(looks, bounds, strict=True), start=1):
            est = _estimate(sample.iloc[:n], y[:n], t[:n], method)
            z = est.estimate / est.se
            row.update({"n_used": n, "estimate": est.estimate, "se": est.se})
            if abs(z) >= c:
                row.update({"stopped_at": k, "rejected": True})
                break
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_trials(trials: pd.DataFrame, effect: float) -> dict[str, float]:
    n = len(trials)
    rej = trials["rejected"].to_numpy(float)
    return {
        "rejection_rate": float(rej.mean()),
        "rejection_se": float(np.sqrt(rej.mean() * (1 - rej.mean()) / n)),
        "mean_members": float(trials["n_used"].mean()),
        "mean_estimate": float(trials["estimate"].mean()),
        "bias": float(trials["estimate"].mean() - effect),
        "stopped_early": float(
            (trials["stopped_at"] < trials["stopped_at"].max()).mean()
        ),
    }
