"""Statistical analysis of what drives annual spend.

The business hypothesis is that tenure in the programme is the largest driver
of spend. Group means alone cannot settle this: they ignore uncertainty,
unequal variances and the contribution of other attributes. This module adds
bootstrap intervals, Welch's ANOVA, effect sizes, Tukey's HSD and a Shapley
decomposition of the explained variance.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

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


def welch_anova(df: pd.DataFrame, group: str, value: str = TARGET) -> dict[str, float]:
    """Welch's one-way ANOVA, valid when group variances differ (Welch, 1951)."""
    parts = [g[value].to_numpy(float) for _, g in df.groupby(group, observed=True)]
    k = len(parts)
    n = np.array([len(p) for p in parts], float)
    means = np.array([p.mean() for p in parts])
    var = np.array([p.var(ddof=1) for p in parts])
    w = n / var
    grand = (w * means).sum() / w.sum()
    a = (w * (means - grand) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k**2 - 1) * lam
    f = a / b
    df2 = (k**2 - 1) / (3 * lam)
    return {
        "f": float(f),
        "df1": float(k - 1),
        "df2": float(df2),
        "p_value": float(stats.f.sf(f, k - 1, df2)),
    }


def effect_sizes(df: pd.DataFrame, group: str, value: str = TARGET) -> dict[str, float]:
    """Eta squared and the less biased omega squared of a one-way layout."""
    y = df[value].to_numpy(float)
    grand = y.mean()
    groups = [g[value].to_numpy(float) for _, g in df.groupby(group, observed=True)]
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((y - grand) ** 2).sum()
    ss_within = ss_total - ss_between
    k, n = len(groups), len(y)
    ms_within = ss_within / (n - k)
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return {"eta_squared": float(ss_between / ss_total), "omega_squared": float(omega)}


def tukey_table(df: pd.DataFrame, group: str, value: str = TARGET) -> pd.DataFrame:
    """All pairwise mean differences with Tukey HSD p-values and intervals."""
    levels, parts = zip(
        *[
            (lvl, g[value].to_numpy(float))
            for lvl, g in df.groupby(group, observed=True)
        ]
    )
    res = stats.tukey_hsd(*parts)
    ci = res.confidence_interval()
    rows = []
    for i, j in combinations(range(len(levels)), 2):
        rows.append(
            {
                "group_a": levels[i],
                "group_b": levels[j],
                "diff": float(parts[i].mean() - parts[j].mean()),
                "ci_low": float(ci.low[i, j]),
                "ci_high": float(ci.high[i, j]),
                "p_value": float(res.pvalue[i, j]),
            }
        )
    return pd.DataFrame(rows)


def _design(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features:
        return np.empty((len(df), 0))
    return pd.get_dummies(df[features], drop_first=True, dtype=float).to_numpy()


def _r2(df: pd.DataFrame, features: list[str], y: np.ndarray) -> float:
    X = _design(df, features)
    if X.shape[1] == 0:
        return 0.0
    return float(LinearRegression().fit(X, y).score(X, y))
