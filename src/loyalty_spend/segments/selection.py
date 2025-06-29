"""Choosing the number of segments.

The average silhouette width (Rousseeuw, 1987) compares, for each member,
the mean distance to its own segment with the mean distance to the nearest
other segment; values near one mean compact, well separated segments and
values near zero mean that the segments overlap. The number of segments
with the largest average width is chosen, and the full table is kept so
that a business preference for fewer segments can be weighed against it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import silhouette_samples

from loyalty_spend.segments.kmedoids import KMedoids


def silhouette_table(
    distances: ArrayLike,
    ks: Iterable[int] = range(2, 9),
    n_init: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    D = np.nan_to_num(np.asarray(distances, float), nan=1.0)
    rows = []
    for k in ks:
        model = KMedoids(k, n_init=n_init, seed=seed).fit(D)
        widths = silhouette_samples(D, model.labels_, metric="precomputed")
        sizes = np.bincount(model.labels_)
        rows.append(
            {
                "k": k,
                "silhouette": float(widths.mean()),
                "worst_segment_silhouette": float(
                    min(widths[model.labels_ == c].mean() for c in range(k))
                ),
                "smallest_segment": int(np.min(sizes)),
                "cost": model.inertia_,
            }
        )
    return pd.DataFrame(rows)


def choose_k(table: pd.DataFrame, min_size: int = 1) -> int:
    """The ``k`` with the largest average silhouette among segmentations
    whose smallest segment has at least ``min_size`` members."""
    eligible = table[table["smallest_segment"] >= min_size]
    if eligible.empty:
        raise ValueError("no segmentation satisfies the minimum segment size")
    return int(eligible.loc[eligible["silhouette"].idxmax(), "k"])
