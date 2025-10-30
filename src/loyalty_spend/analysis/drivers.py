"""Statistical analysis of what drives annual spend.

The business hypothesis is that tenure in the programme is the largest driver
of spend. Group means alone cannot settle this: they ignore uncertainty,
unequal variances and the contribution of other attributes. This module adds
bootstrap intervals, Welch's ANOVA, effect sizes, Tukey's HSD, the
Games-Howell procedure and a Shapley decomposition of the explained
variance.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

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


def games_howell(
    df: pd.DataFrame,
    group: str,
    value: str = TARGET,
    alpha: float = 0.05,
    intervals: bool = True,
) -> pd.DataFrame:
    r"""Pairwise comparisons that do not assume equal variances.

    Tukey's HSD pools the variance of all groups, which misstates the error
    of comparisons between groups whose spreads differ. Games and Howell
    (1976) use the standard error of each pair,
    :math:`\sqrt{s_i^2/n_i + s_j^2/n_j}`, Welch-Satterthwaite degrees of
    freedom and the studentized range distribution for :math:`k` groups.
    Quantiles of that distribution are slow to compute, so ``intervals``
    can be switched off when only p-values are needed.
    """
    levels, parts = zip(
        *[
            (lvl, g[value].to_numpy(float))
            for lvl, g in df.groupby(group, observed=True)
        ]
    )
    k = len(parts)
    rows = []
    for i, j in combinations(range(k), 2):
        a, b = parts[i], parts[j]
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
        diff = a.mean() - b.mean()
        q = abs(diff) / se * np.sqrt(2)
        crit = (
            stats.studentized_range.ppf(1 - alpha, k, dof) / np.sqrt(2)
            if intervals
            else np.nan
        )
        rows.append(
            {
                "group_a": levels[i],
                "group_b": levels[j],
                "diff": float(diff),
                "se": float(se),
                "df": float(dof),
                "ci_low": float(diff - crit * se),
                "ci_high": float(diff + crit * se),
                "p_value": float(stats.studentized_range.sf(q, k, dof)),
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


def lmg_importance(
    df: pd.DataFrame, features: list[str], value: str = TARGET
) -> pd.DataFrame:
    """Shapley decomposition of the OLS R squared over features.

    Each feature receives its average increase in R squared over all orders
    of entry (Lindeman, Merenda and Gold, 1980; Groemping, 2006). The shares
    add up to the R squared of the full model, so they answer "which driver
    explains most of the variance" without depending on the order in which
    variables are added. Categorical features enter as dummy blocks.
    """
    data = df[features].dropna()
    y = df.loc[data.index, value].to_numpy(float)
    p = len(features)
    cache: dict[frozenset, float] = {}

    def r2(subset: frozenset) -> float:
        if subset not in cache:
            cache[subset] = _r2(data, [f for f in features if f in subset], y)
        return cache[subset]

    rows = []
    for f in features:
        others = [g for g in features if g != f]
        total = 0.0
        for size in range(p):
            weight = factorial(size) * factorial(p - size - 1) / factorial(p)
            for subset in combinations(others, size):
                s = frozenset(subset)
                total += weight * (r2(s | {f}) - r2(s))
        rows.append({"feature": f, "r2_share": total})
    out = pd.DataFrame(rows).sort_values("r2_share", ascending=False)
    out["relative_share"] = out["r2_share"] / out["r2_share"].sum()
    return out.reset_index(drop=True)


def _rss(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual sums of squares of y (one column per permutation) on X."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return ((y - X @ beta) ** 2).sum(axis=0)


def freedman_lane(
    df: pd.DataFrame,
    test: list[str],
    covariates: list[str],
    value: str = TARGET,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Permutation test of ``test`` attributes given ``covariates``.

    The partial F statistic for adding the tested attributes to a linear
    model with the covariates is referred to its permutation distribution
    under the procedure of Freedman and Lane (1983): residuals of the
    reduced model are permuted and added back to its fitted values, which
    keeps the covariate effects intact under the null hypothesis. Unlike
    the F distribution, this does not assume normal errors.
    """
    data = df[[*test, *covariates, value]].dropna()
    y = data[value].to_numpy(float)
    ones = np.ones((len(y), 1))
    Xr = np.hstack([ones, _design(data, covariates)])
    Xf = np.hstack([Xr, _design(data, test)])
    df1 = np.linalg.matrix_rank(Xf) - np.linalg.matrix_rank(Xr)
    df2 = len(y) - np.linalg.matrix_rank(Xf)
    beta, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    fitted, resid = Xr @ beta, y - Xr @ beta

    def f_stat(ys: np.ndarray) -> np.ndarray:
        rr, rf = _rss(Xr, ys), _rss(Xf, ys)
        return ((rr - rf) / df1) / (rf / df2)

    observed = float(f_stat(y[:, None])[0])
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_perm, 200):
        m = min(200, n_perm - start)
        perm = np.stack([rng.permutation(resid) for _ in range(m)], axis=1)
        exceed += int((f_stat(fitted[:, None] + perm) >= observed).sum())
    return {
        "f": observed,
        "df1": float(df1),
        "df2": float(df2),
        "p_permutation": (exceed + 1) / (n_perm + 1),
        "p_f": float(stats.f.sf(observed, df1, df2)),
    }
