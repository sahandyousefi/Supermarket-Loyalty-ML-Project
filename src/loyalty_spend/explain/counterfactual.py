"""What would move a member into the next reward tier?

For a member below a threshold, the counterfactual asks for the smallest
change of one actionable attribute (by default the first-month spend, the
only amount a member controls) that lifts the forecast to the next
threshold. The search is a bisection on the size of the change, which is
exact for a forecast that is monotone in the attribute and is checked by
re-scoring the result otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from loyalty_spend.business.rewards import RewardSchedule

Predict = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class TierCounterfactual:
    prediction: float
    next_threshold: float | None
    gap: float
    feature: str
    current_value: float
    required_value: float | None

    @property
    def feasible(self) -> bool:
        return self.required_value is not None

    @property
    def required_change(self) -> float | None:
        return (
            None
            if self.required_value is None
            else self.required_value - self.current_value
        )


def next_tier_counterfactual(
    predict: Predict,
    row: pd.DataFrame,
    schedule: RewardSchedule,
    feature: str = "first_month",
    upper: float | None = None,
    tol: float = 1e-3,
) -> TierCounterfactual:
    """Smallest value of ``feature`` that lifts the forecast to the next threshold.

    ``upper`` bounds the search (for example the largest value seen in the
    training data); without a feasible value inside it, ``required_value``
    is ``None``.
    """
    if len(row) != 1:
        raise ValueError("pass exactly one member")
    pred = float(predict(row)[0])
    thresholds = [t for t, _ in schedule.tiers if t > pred]
    current = float(row[feature].iloc[0])
    if not thresholds:
        return TierCounterfactual(pred, None, 0.0, feature, current, current)
    target = thresholds[0]

    def score(v: float) -> float:
        return float(predict(row.assign(**{feature: v}))[0])

    hi = upper if upper is not None else max(current * 10, current + 1000)
    if score(hi) < target:
        return TierCounterfactual(pred, target, target - pred, feature, current, None)
    lo = current
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if score(mid) >= target:
            hi = mid
        else:
            lo = mid
    return TierCounterfactual(pred, target, target - pred, feature, current, hi)


IMMUTABLE = ("region", "loyalty_years", "joining_month", "promotion")


def minimal_change_counterfactual(
    predict: Predict,
    row: pd.DataFrame,
    target: float,
    reference: pd.DataFrame,
    actionable: tuple[str, ...] = ("first_month", "items_in_first_month"),
    n_grid: int = 41,
    quantiles: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Plausible changes of the actionable attributes that reach ``target``.

    Attributes outside ``actionable`` (region, tenure, joining month and
    promotion) never change. Each actionable attribute moves on a grid
    inside its 1st to 99th percentile in ``reference``, so the answer never
    asks for a basket nobody has. Changes are measured in units of each
    attribute's median absolute deviation, so a change of one item and of a
    few currency units are comparable (Wachter et al., 2018). Returns the
    feasible combinations, cheapest first.
    """
    import itertools

    if any(a in IMMUTABLE for a in actionable):
        raise ValueError("immutable attributes cannot be changed")
    grids, scales = [], []
    for a in actionable:
        values = reference[a].dropna().astype(float)
        lo, hi = values.quantile(list(quantiles))
        grids.append(np.linspace(lo, hi, n_grid))
        mad = float(np.median(np.abs(values - values.median())))
        scales.append(mad if mad > 0 else float(values.std()) or 1.0)
    combos = np.array(list(itertools.product(*grids)))
    candidates = pd.concat([row] * len(combos), ignore_index=True)
    for j, a in enumerate(actionable):
        candidates[a] = combos[:, j]
    scores = predict(candidates)
    current = np.array([float(row[a].iloc[0]) for a in actionable])
    cost = (np.abs(combos - current) / np.array(scales)).sum(axis=1)
    out = pd.DataFrame(combos, columns=list(actionable))
    out["forecast"] = scores
    out["cost"] = cost
    return out[out["forecast"] >= target].sort_values("cost").reset_index(drop=True)
