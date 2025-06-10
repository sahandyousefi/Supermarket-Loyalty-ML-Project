import numpy as np
import pytest

from loyalty_spend.uncertainty.conformal import conformal_quantile
from loyalty_spend.uncertainty.weighted import weighted_conformal_quantile


def test_equal_weights_reproduce_split_conformal():
    rng = np.random.default_rng(0)
    for n in (9, 50, 333):
        s = rng.exponential(size=n)
        for alpha in (0.05, 0.1, 0.2):
            q = weighted_conformal_quantile(s, np.ones(n), [1.0], alpha)[0]
            assert q == pytest.approx(conformal_quantile(s, alpha))


def test_large_test_weight_gives_an_infinite_interval():
    q = weighted_conformal_quantile([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [10.0], 0.1)
    assert np.isinf(q[0])


def test_weights_move_the_quantile_towards_heavy_scores():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    light = weighted_conformal_quantile(s, [1, 1, 1, 0.1], [0.1], 0.3)[0]
    heavy = weighted_conformal_quantile(s, [0.1, 1, 1, 1], [0.1], 0.3)[0]
    assert light < heavy


def test_invalid_weights():
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0, 2.0], [1.0], [1.0], 0.1)
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0], [-1.0], [1.0], 0.1)
