"""Spend forecasts and reward cost by segment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.business.breakdown import cost_by_group
from loyalty_spend.business.rewards import LiabilityForecast


def liability_by_segment(
    forecast: LiabilityForecast, labels: ArrayLike
) -> pd.DataFrame:
    """Aggregate a Monte Carlo reward forecast over segments."""
    return cost_by_group(forecast, np.asarray(labels, int), name="segment")
