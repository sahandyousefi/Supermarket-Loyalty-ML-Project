import numpy as np
import pytest

from loyalty_spend.monitoring.drift import numeric_psi, psi_from_counts, quantile_edges


def test_identical_counts_have_zero_psi():
    assert psi_from_counts([10, 20, 30], [1, 2, 3]) == pytest.approx(0.0)


def test_psi_is_symmetric_and_positive():
    a, b = [50, 30, 20], [20, 30, 50]
    assert psi_from_counts(a, b) == pytest.approx(psi_from_counts(b, a))
    assert psi_from_counts(a, b) > 0.25


def test_psi_grows_with_the_size_of_a_mean_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    values = [numeric_psi(ref, rng.normal(loc=m, size=5000)) for m in (0, 0.2, 0.5, 1)]
    assert values[0] < 0.02
    assert values == sorted(values)
    assert values[-1] > 0.25


def test_tied_values_give_fewer_bins():
    edges = quantile_edges(np.repeat([0, 1, 2], 100), 10)
    assert list(edges) == [0.0, 1.0, 2.0]


def test_missing_values_count_as_drift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=2000)
    cur = ref.copy()
    cur[:400] = np.nan
    assert numeric_psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    assert numeric_psi(ref, cur) > 0.1
