"""Fidelity of synthetic data: marginals and pairwise dependence.

Marginals are compared with the Kolmogorov-Smirnov statistic and the
Wasserstein distance scaled by the real interquartile range for amounts,
and with the total variation distance for categories. Pairwise dependence
uses one measure per kind of pair: Spearman's rho for two amounts,
Cramer's V (bias corrected, Bergsma, 2013) for two categories and the
correlation ratio eta for an amount and a category. Each is computed on
the real and the synthetic data and reported with the absolute difference.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not isinstance(
        s.dtype, pd.CategoricalDtype
    )


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a.astype(str), b.astype(str)).to_numpy()
    n = table.sum()
    r, k = table.shape
    if r < 2 or k < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def correlation_ratio(values: pd.Series, groups: pd.Series) -> float:
    keep = values.notna() & groups.notna()
    v = values[keep].to_numpy(float)
    g = groups[keep].astype(str).to_numpy()
    total = np.sum((v - v.mean()) ** 2)
    if total == 0:
        return 0.0
    between = sum(
        (g == lv).sum() * (v[g == lv].mean() - v.mean()) ** 2 for lv in np.unique(g)
    )
    return float(np.sqrt(between / total))


def marginal_fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in real.columns:
        r, s = real[col], synth[col]
        if _is_numeric(r):
            r_obs, s_obs = r.dropna().to_numpy(float), s.dropna().to_numpy(float)
            iqr = float(np.subtract(*np.percentile(r_obs, [75, 25]))) or 1.0
            rows.append(
                {
                    "column": col,
                    "measure": "ks",
                    "value": float(stats.ks_2samp(r_obs, s_obs).statistic),
                    "wasserstein_iqr": stats.wasserstein_distance(r_obs, s_obs) / iqr,
                    "missing_real": float(r.isna().mean()),
                    "missing_synth": float(s.isna().mean()),
                }
            )
        else:
            p = r.astype(str).where(r.notna(), "missing").value_counts(normalize=True)
            q = s.astype(str).where(s.notna(), "missing").value_counts(normalize=True)
            p, q = p.align(q, fill_value=0.0)
            rows.append(
                {
                    "column": col,
                    "measure": "tvd",
                    "value": float(0.5 * (p - q).abs().sum()),
                    "wasserstein_iqr": np.nan,
                    "missing_real": float(r.isna().mean()),
                    "missing_synth": float(s.isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def _association(frame: pd.DataFrame, a: str, b: str) -> tuple[str, float]:
    x, y = frame[a], frame[b]
    if _is_numeric(x) and _is_numeric(y):
        keep = x.notna() & y.notna()
        rho = stats.spearmanr(x[keep], y[keep]).statistic
        return "spearman", float(rho)
    if not _is_numeric(x) and not _is_numeric(y):
        return "cramers_v", cramers_v(x, y)
    num, cat = (x, y) if _is_numeric(x) else (y, x)
    return "eta", correlation_ratio(num, cat)


def pairwise_fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(real.columns, 2):
        measure, r = _association(real, a, b)
        _, s = _association(synth, a, b)
        rows.append(
            {
                "pair": f"{a} | {b}",
                "measure": measure,
                "real": r,
                "synthetic": s,
                "difference": abs(r - s),
            }
        )
    return pd.DataFrame(rows).sort_values("difference", ascending=False)


def fidelity_summary(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    marg = marginal_fidelity(real, synth)
    pair = pairwise_fidelity(real, synth)
    return {
        "max_ks": float(marg.loc[marg["measure"] == "ks", "value"].max()),
        "max_tvd": float(marg.loc[marg["measure"] == "tvd", "value"].max()),
        "mean_pair_difference": float(pair["difference"].mean()),
        "max_pair_difference": float(pair["difference"].max()),
    }
