"""Segment stability by bootstrap (Hennig, 2007).

A segment that only exists in one sample is not a segment of the customer
base. Each bootstrap resample is clustered again; for every original segment
the Jaccard similarity with its best-matching resampled segment is computed
on the members present in the resample. The mean over resamples measures
stability: Hennig suggests that values above 0.75 indicate a stable segment
and values below 0.5 a segment that dissolves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.segments.kmedoids import KMedoids


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else 0.0


def bootstrap_stability(
    distances: ArrayLike,
    labels: ArrayLike,
    n_boot: int = 50,
    n_init: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean and minimum bootstrap Jaccard similarity of every segment."""
    D = np.nan_to_num(np.asarray(distances, float), nan=1.0)
    labels = np.asarray(labels, int)
    k = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    scores = np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = np.unique(rng.integers(0, len(D), size=len(D)))
        sub = KMedoids(k, n_init=n_init, seed=seed + b).fit(D[np.ix_(idx, idx)])
        for c in range(k):
            original = labels[idx] == c
            if not original.any():
                continue
            scores[b, c] = max(jaccard(original, sub.labels_ == j) for j in range(k))
    return pd.DataFrame(
        {
            "segment": np.arange(k),
            "size": np.bincount(labels, minlength=k),
            "mean_jaccard": scores.mean(axis=0),
            "min_jaccard": scores.min(axis=0),
            "stable": scores.mean(axis=0) > 0.75,
        }
    )
