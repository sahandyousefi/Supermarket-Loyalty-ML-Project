"""Reward cost broken down by any grouping of customers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.business.rewards import LiabilityForecast


def cost_by_group(
    forecast: LiabilityForecast, groups: ArrayLike, name: str = "group"
) -> pd.DataFrame:
    """Expected and plug-in cost, spend and tier uncertainty per group.

    Expected rewards add up over customers, so each group's share of the
    expected total is exact. The Jensen gap is the difference between the
    expected cost and the cost of the expected spend; it is largest where
    many customers sit near a threshold.
    """
    pc = forecast.per_customer.assign(**{name: np.asarray(groups)})
    out = pc.groupby(name, observed=True).agg(
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
