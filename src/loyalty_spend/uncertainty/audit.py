"""Audit of conditional coverage across subgroups.

Conformal intervals guarantee coverage on average over members, not within
a region or tenure band. For every level of every audited attribute the
audit counts how many members' spend falls inside their interval, gives an
exact Clopper-Pearson interval for the coverage, and tests the nominal
level with an exact two-sided binomial test. P-values are adjusted by
Benjamini-Hochberg over all levels, since an audit over many levels would
otherwise flag some by chance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from loyalty_spend.uncertainty.conformal import Intervals


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2
    lo = stats.beta.ppf(tail, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - tail, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def benjamini_hochberg(p: ArrayLike) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def coverage_audit(
    intervals: Intervals,
    y: ArrayLike,
    frame: pd.DataFrame,
    columns: Sequence[str],
    alpha: float = 0.1,
    level: float = 0.95,
    fdr: float = 0.05,
) -> pd.DataFrame:
    y = np.asarray(y, float)
    covered = (y >= intervals.lower) & (y <= intervals.upper)
    width = intervals.width
    rows = []
    for col in columns:
        values = frame[col].astype(str).to_numpy()
        for lv in sorted(set(values)):
            sel = values == lv
            n, k = int(sel.sum()), int(covered[sel].sum())
            lo, hi = clopper_pearson(k, n, level)
            rows.append(
                {
                    "attribute": col,
                    "level": lv,
                    "n": n,
                    "coverage": k / n,
                    "ci_low": lo,
                    "ci_high": hi,
                    "mean_width": float(width[sel].mean()),
                    "p_value": float(stats.binomtest(k, n, 1 - alpha).pvalue),
                }
            )
    table = pd.DataFrame(rows)
    table["p_adjusted"] = benjamini_hochberg(table["p_value"])
    table["flagged"] = table["p_adjusted"] < fdr
    return table
