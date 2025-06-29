"""Spend forecasts and reward cost by segment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.business.rewards import LiabilityForecast


def liability_by_segment(
    forecast: LiabilityForecast, labels: ArrayLike
) -> pd.DataFrame:
    """Aggregate a Monte Carlo reward forecast over segments.

    Expected rewards add up over customers, so each segment's share of the
    expected total is exact; the uncertainty of each segment's total would
    need the joint draws and is not reported here.
    """
    pc = forecast.per_customer.assign(segment=np.asarray(labels, int))
    out = pc.groupby("segment").agg(
        customers=("customer_id", "size"),
        mean_predicted_spend=("predicted_spend", "mean"),
        expected_reward=("expected_reward", "sum"),
        plug_in_reward=("plug_in_reward", "sum"),
        mean_tier_change_probability=("tier_change_probability", "mean"),
    )
    out["share_of_expected_cost"] = (
        out["expected_reward"] / out["expected_reward"].sum()
    )
    out["jensen_gap"] = out["expected_reward"] - out["plug_in_reward"]
    return out.reset_index()
