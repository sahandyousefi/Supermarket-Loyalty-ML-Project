import numpy as np
import pytest

from loyalty_spend.data.io import load_clean
from loyalty_spend.data.schema import CATEGORICAL, allowed_values
from loyalty_spend.monitoring.drift import (
    category_psi,
    numeric_psi,
    psi_from_counts,
    quantile_edges,
    unseen_levels,
)


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


def test_category_psi_detects_a_change_in_mix():
    ref = ["Europe"] * 50 + ["Americas"] * 50
    assert category_psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    assert category_psi(ref, ["Europe"] * 90 + ["Americas"] * 10) > 0.25


def test_unseen_levels_and_missing_values_register():
    ref = ["a", "b"] * 100
    cur = ["a", "b"] * 80 + ["c"] * 40
    assert unseen_levels(ref, cur) == ["c"]
    assert category_psi(ref, cur, levels=["a", "b"]) > 0.1
    with_missing = ["a", "b"] * 80 + [None] * 40
    assert category_psi(ref, with_missing) > 0.1


def test_train_and_test_files_show_no_large_shift(train_df, raw_dir):
    # With 250 scoring rows and up to 13 levels, sampling noise alone gives a
    # PSI of several hundredths, so only the large-shift threshold is checked.
    test, _ = load_clean(raw_dir / "test.csv", impute=False, require_target=False)
    for col in CATEGORICAL:
        assert category_psi(train_df[col], test[col], allowed_values(col)) < 0.25
