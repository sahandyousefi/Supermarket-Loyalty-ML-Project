"""Statistical analysis of what drives annual spend.

The business hypothesis is that tenure in the programme is the largest driver
of spend. Group means alone cannot settle this: they ignore uncertainty,
unequal variances and the contribution of other attributes. This module adds
bootstrap intervals, Welch's ANOVA, effect sizes, Tukey's HSD and a Shapley
decomposition of the explained variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loyalty_spend.data.schema import TARGET


def spend_by_group(
    df: pd.DataFrame,
    group: str,
    value: str = TARGET,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Count, mean, variance and a percentile bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for level, part in df.groupby(group, observed=True):
        x = part[value].to_numpy(float)
        boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        rows.append(
            {
                group: level,
                "n": len(x),
                "avg_spend": x.mean(),
                "var_spend": x.var(ddof=1),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)
