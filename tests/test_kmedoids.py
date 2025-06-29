import numpy as np
import pytest

from loyalty_spend.segments.kmedoids import KMedoids


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [6, 0], [0, 6]])
    points = np.vstack([c + rng.normal(size=(60, 2)) for c in centres])
    truth = np.repeat([0, 1, 2], 60)
    D = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    return D, truth


def test_recovers_well_separated_clusters(blobs):
    D, truth = blobs
    model = KMedoids(3, seed=1).fit(D)
    # Each true cluster maps onto one found cluster, up to a stray point
    # from the overlapping tails.
    majority = [np.bincount(model.labels_[truth == c]).argmax() for c in range(3)]
    assert sorted(majority) == [0, 1, 2]
    agreement = np.mean(model.labels_ == np.array(majority)[truth])
    assert agreement > 0.97
    assert all(model.labels_[m] == i for i, m in enumerate(model.medoids_))


def test_more_clusters_never_cost_more(blobs):
    D, _ = blobs
    costs = [KMedoids(k, seed=2).fit(D).inertia_ for k in (1, 2, 3, 4)]
    assert costs == sorted(costs, reverse=True)


def test_invalid_input():
    with pytest.raises(ValueError):
        KMedoids(0)
    with pytest.raises(ValueError):
        KMedoids(2).fit(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        KMedoids(5).fit(np.zeros((3, 3)))
