import numpy as np
import pytest

from loyalty_spend.segments.selection import choose_k, silhouette_table


def _blobs(k, seed=0):
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(k) / k
    centres = 12 * np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.vstack([c + rng.normal(size=(40, 2)) for c in centres])
    return np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))


@pytest.mark.parametrize("k", [3, 4])
def test_silhouette_finds_the_number_of_blobs(k):
    table = silhouette_table(_blobs(k, seed=k), ks=range(2, 7), n_init=5)
    assert choose_k(table) == k
    assert table["silhouette"].between(-1, 1).all()


def test_minimum_size_constraint():
    table = silhouette_table(_blobs(3), ks=range(2, 5), n_init=3)
    assert choose_k(table, min_size=10) in (2, 3, 4)
    with pytest.raises(ValueError):
        choose_k(table, min_size=10_000)
