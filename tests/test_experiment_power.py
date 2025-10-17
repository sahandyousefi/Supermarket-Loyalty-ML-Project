import numpy as np
import pytest

from loyalty_spend.experiment.power import (
    minimum_detectable_effect,
    power_two_sample,
    sample_size,
)


def test_textbook_sample_size():
    # Cohen's d = 0.5 needs 64 per group for 80% power at the 5% level.
    assert sample_size(0.5, 1.0) == (64, 64)
    assert power_two_sample(0.5, 1.0, 64, 64) == pytest.approx(0.8015, abs=1e-3)


def test_power_matches_simulation():
    rng = np.random.default_rng(0)
    from scipy import stats

    hits = 0
    for _ in range(4000):
        a, b = rng.normal(0.3, 1, 40), rng.normal(0, 1, 60)
        hits += stats.ttest_ind(a, b).pvalue < 0.05
    assert hits / 4000 == pytest.approx(power_two_sample(0.3, 1.0, 40, 60), abs=0.02)


def test_adjustment_shrinks_the_required_sample():
    n_raw = sample_size(1.0, 26.8)[1]
    n_adj = sample_size(1.0, 26.8, r2=0.99, n_covariates=20)[1]
    assert n_adj < n_raw / 50


def test_minimum_detectable_effect_inverts_power():
    mde = minimum_detectable_effect(2.0, 100, 150, power=0.9)
    assert power_two_sample(mde, 2.0, 100, 150) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        power_two_sample(1.0, 1.0, 5, 5, r2=1.0)
