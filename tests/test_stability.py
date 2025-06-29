import numpy as np

from loyalty_spend.segments.kmedoids import KMedoids
from loyalty_spend.segments.stability import bootstrap_stability, jaccard


def _points(centres, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.vstack([np.asarray(c) + rng.normal(size=(50, 2)) for c in centres])
    return np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))


def test_jaccard():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert jaccard(a, b) == 1 / 3
    assert jaccard(a & False, b & False) == 0.0


def test_real_segments_are_stable_and_forced_splits_are_not():
    separated = _points([[0, 0], [10, 0]])
    labels = KMedoids(2, seed=0).fit(separated).labels_
    table = bootstrap_stability(separated, labels, n_boot=15)
    assert table["stable"].all()

    single_blob = _points([[0, 0], [0, 0]], seed=1)
    labels = KMedoids(3, seed=0).fit(single_blob).labels_
    table = bootstrap_stability(single_blob, labels, n_boot=15)
    assert table["mean_jaccard"].min() < 0.75
