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

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

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
