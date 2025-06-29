"""K-medoids clustering on a precomputed distance matrix.

Gower distances have no mean, so k-means does not apply; k-medoids uses
actual members as cluster centres, which also makes each segment easy to
describe by its most typical member. The alternating algorithm of Park and
Jun (2009) assigns members to the nearest medoid and then moves each medoid
to the member with the smallest total distance within its cluster, until
nothing changes. Several restarts from k-means++ style seeds guard against
poor local optima.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _seed_medoids(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(D)
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        nearest = D[:, medoids].min(axis=1)
        p = nearest**2
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1 / n)
        medoids.append(int(rng.choice(n, p=p)))
    return np.array(medoids)


def _alternate(
    D: np.ndarray, medoids: np.ndarray, max_iter: int
) -> tuple[np.ndarray, np.ndarray, float]:
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new = medoids.copy()
        for c in range(len(medoids)):
            members = np.flatnonzero(labels == c)
            if members.size:
                within = D[np.ix_(members, members)].sum(axis=1)
                new[c] = members[np.argmin(within)]
        if np.array_equal(np.sort(new), np.sort(medoids)):
            break
        medoids = new
    labels = np.argmin(D[:, medoids], axis=1)
    cost = float(D[np.arange(len(D)), medoids[labels]].sum())
    return medoids, labels, cost


class KMedoids:
    """K-medoids with restarts; ``fit`` takes a square distance matrix."""

    def __init__(
        self, n_clusters: int, n_init: int = 10, max_iter: int = 100, seed: int = 0
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, distances: ArrayLike) -> KMedoids:
        D = np.asarray(distances, float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("distances must be a square matrix")
        if self.n_clusters > len(D):
            raise ValueError("more clusters than members")
        D = np.nan_to_num(D, nan=np.nanmax(D))
        rng = np.random.default_rng(self.seed)
        best: tuple[np.ndarray, np.ndarray, float] | None = None
        for _ in range(self.n_init):
            result = _alternate(
                D, _seed_medoids(D, self.n_clusters, rng), self.max_iter
            )
            if best is None or result[2] < best[2]:
                best = result
        assert best is not None
        medoids, labels, cost = best
        # Order clusters by size so that cluster 0 is the largest segment.
        sizes = np.bincount(labels, minlength=self.n_clusters)
        order = np.argsort(-sizes, kind="stable")
        relabel = np.empty_like(order)
        relabel[order] = np.arange(len(order))
        self.medoids_ = medoids[order]
        self.labels_ = relabel[labels]
        self.inertia_ = cost
        return self
