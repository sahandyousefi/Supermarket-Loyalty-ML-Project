r"""Distribution drift between a reference sample and a new sample.

The population stability index of a binned variable is

.. math::

    \mathrm{PSI} = \sum_{b} (a_b - e_b) \log \frac{a_b}{e_b},

where :math:`e_b` and :math:`a_b` are the shares of the reference and the new
sample in bin :math:`b`. It is the symmetrised Kullback-Leibler divergence of
the two binned distributions. The usual reading is below 0.1 stable, 0.1 to
0.25 a moderate shift and above 0.25 a large shift (Siddiqi, 2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from loyalty_spend.data.schema import CATEGORICAL, NUMERIC

EPS = 1e-4


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    shares = counts / total if total else np.zeros_like(counts, dtype=float)
    # A small floor keeps the log finite for bins that one sample leaves empty.
    shares = np.maximum(shares, EPS)
    return shares / shares.sum()


def psi_from_counts(expected: ArrayLike, actual: ArrayLike) -> float:
    """PSI of two count vectors over the same bins."""
    e = _shares(np.asarray(expected, float))
    a = _shares(np.asarray(actual, float))
    return float(np.sum((a - e) * np.log(a / e)))


def quantile_edges(reference: ArrayLike, n_bins: int = 10) -> np.ndarray:
    """Interior bin edges at the reference quantiles, without duplicates.

    Heavily tied variables (for example item counts) produce fewer than
    ``n_bins`` bins rather than empty ones.
    """
    x = pd.Series(reference, dtype=float).dropna().to_numpy()
    if x.size == 0:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(x, qs))


def numeric_psi(reference: ArrayLike, current: ArrayLike, n_bins: int = 10) -> float:
    """PSI of a numeric variable with quantile bins from the reference.

    Missing values form a bin of their own, so a rise in missingness counts
    as drift.
    """
    ref = pd.Series(reference, dtype=float)
    cur = pd.Series(current, dtype=float)
    edges = quantile_edges(ref, n_bins)

    def counts(s: pd.Series) -> np.ndarray:
        x = s.dropna().to_numpy()
        binned = np.bincount(
            np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1
        )
        return np.append(binned, s.isna().sum())

    return psi_from_counts(counts(ref), counts(cur))


def category_psi(
    reference: ArrayLike, current: ArrayLike, levels: list[str] | None = None
) -> float:
    """PSI of a categorical variable.

    Bins are the declared ``levels`` (or the reference levels), one bin for
    missing values and one for levels that the reference never showed, so a
    new region or a typo in an extract registers as drift instead of being
    dropped.
    """
    ref = pd.Series(reference, dtype=object)
    cur = pd.Series(current, dtype=object)
    known = (
        list(levels)
        if levels is not None
        else sorted(ref.dropna().astype(str).unique())
    )

    def counts(s: pd.Series) -> np.ndarray:
        text = s.dropna().astype(str)
        known_counts = [int((text == lvl).sum()) for lvl in known]
        unseen = int((~text.isin(known)).sum())
        return np.array([*known_counts, unseen, int(s.isna().sum())])

    return psi_from_counts(counts(ref), counts(cur))


def unseen_levels(reference: ArrayLike, current: ArrayLike) -> list[str]:
    """Levels present in ``current`` but never in ``reference``."""
    ref = set(pd.Series(reference, dtype=object).dropna().astype(str))
    cur = set(pd.Series(current, dtype=object).dropna().astype(str))
    return sorted(cur - ref)


def ks_test(reference: ArrayLike, current: ArrayLike) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value, ignoring missing values."""
    ref = pd.Series(reference, dtype=float).dropna()
    cur = pd.Series(current, dtype=float).dropna()
    if ref.empty or cur.empty:
        return float("nan"), float("nan")
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def chi_square_test(reference: ArrayLike, current: ArrayLike) -> tuple[float, float]:
    """Chi-square test of homogeneity of two categorical samples.

    Levels seen in neither sample are dropped; missing values are a level, as
    in the PSI. Returns the statistic and p-value.
    """
    ref = pd.Series(reference, dtype=object).fillna("<missing>").astype(str)
    cur = pd.Series(current, dtype=object).fillna("<missing>").astype(str)
    table = pd.crosstab(
        np.r_[np.zeros(len(ref)), np.ones(len(cur))],
        np.r_[ref.to_numpy(), cur.to_numpy()],
    )
    if table.shape[1] < 2:
        return 0.0, 1.0
    stat, p, _, _ = stats.chi2_contingency(table.to_numpy())
    return float(stat), float(p)


def benjamini_hochberg(p_values: ArrayLike) -> np.ndarray:
    """Adjusted p-values controlling the false discovery rate (BH, 1995).

    Missing p-values stay missing and do not count towards the number of
    tests.
    """
    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    values = np.empty(m)
    values[order] = adjusted
    out[ok] = values
    return out


@dataclass(frozen=True)
class DriftThresholds:
    """PSI bands and the false discovery rate for the tests."""

    psi_moderate: float = 0.1
    psi_large: float = 0.25
    fdr: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.psi_moderate < self.psi_large:
            raise ValueError("need 0 < psi_moderate < psi_large")
        if not 0 < self.fdr < 1:
            raise ValueError("fdr must be in (0, 1)")


def _status(psi: float, p_adj: float, t: DriftThresholds) -> str:
    # PSI says how large a shift is; the test says whether it is more than
    # sampling noise. A shift is flagged only when both agree.
    significant = bool(p_adj < t.fdr) if not np.isnan(p_adj) else False
    if psi >= t.psi_large and significant:
        return "large"
    if psi >= t.psi_moderate and significant:
        return "moderate"
    return "stable"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    n_bins: int = 10,
    thresholds: DriftThresholds | None = None,
) -> pd.DataFrame:
    """Per-feature drift of ``current`` against ``reference``.

    Numeric features get the quantile-bin PSI and the KS test, categorical
    features the level PSI and the chi-square test. P-values are adjusted
    across all features with Benjamini-Hochberg.
    """
    t = thresholds or DriftThresholds()
    numeric = list(NUMERIC if numeric is None else numeric)
    categorical = list(CATEGORICAL if categorical is None else categorical)
    rows = []
    for col in numeric:
        stat, p = ks_test(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "kind": "numeric",
                "psi": numeric_psi(reference[col], current[col], n_bins),
                "test": "ks",
                "statistic": stat,
                "p_value": p,
                "unseen": "",
            }
        )
    for col in categorical:
        stat, p = chi_square_test(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "kind": "category",
                "psi": category_psi(reference[col], current[col]),
                "test": "chi2",
                "statistic": stat,
                "p_value": p,
                "unseen": ", ".join(unseen_levels(reference[col], current[col])),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_adjusted"] = benjamini_hochberg(table["p_value"])
    table["status"] = [
        _status(psi, p, t) for psi, p in zip(table["psi"], table["p_adjusted"])
    ]
    return table
