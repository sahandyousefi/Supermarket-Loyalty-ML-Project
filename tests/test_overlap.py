import numpy as np
import pytest

from loyalty_spend.causal.overlap import (
    crump_threshold,
    effective_sample_size,
    overlap_summary,
    trim_mask,
)


def test_effective_sample_size():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert effective_sample_size([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_sample_size(np.zeros(3)) == 0.0


def test_good_overlap_needs_no_trimming():
    e = np.random.default_rng(0).uniform(0.3, 0.7, size=1000)
    assert crump_threshold(e) == 0.0
    assert trim_mask(e, 0.0).all()


def test_poor_overlap_is_trimmed():
    rng = np.random.default_rng(1)
    e = np.r_[rng.uniform(0.2, 0.8, size=900), rng.uniform(0.001, 0.02, size=100)]
    alpha = crump_threshold(e)
    assert 0.02 < alpha < 0.2
    assert trim_mask(e, alpha).sum() >= 880
    with pytest.raises(ValueError):
        trim_mask(e, 0.6)


def test_overlap_summary_reports_ess_loss():
    rng = np.random.default_rng(2)
    t = rng.integers(0, 2, size=2000)
    flat = overlap_summary(np.full(2000, 0.5), t)
    skewed = overlap_summary(rng.beta(0.5, 0.5, size=2000), t)
    assert flat["ess_treated"] == pytest.approx(flat["n_treated"])
    assert skewed["ess_treated"] < 0.8 * skewed["n_treated"]
    assert skewed["share_extreme"] > flat["share_extreme"]
