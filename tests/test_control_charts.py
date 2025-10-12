import numpy as np
import pytest

from loyalty_spend.monitoring.control_charts import (
    cusum_chart,
    cusum_interval,
    ewma_chart,
    hwma_chart,
    mhwma_chart,
    siegmund_arl,
)


def test_ewma_recursion_and_limits():
    x = np.array([1.0, -0.5, 2.0])
    res = ewma_chart(x, lam=0.2, L=3.0)
    z1 = 0.2
    z2 = 0.2 * -0.5 + 0.8 * z1
    z3 = 0.2 * 2.0 + 0.8 * z2
    assert np.allclose(res.statistic, [z1, z2, z3])
    var1 = 0.2 / 1.8 * (1 - 0.8**2)
    assert res.upper[0] == pytest.approx(3 * np.sqrt(var1))
    assert np.allclose(res.lower, -res.upper)


def test_exact_limits_approach_the_asymptotic_ones():
    res = ewma_chart(np.zeros(200), lam=0.1, L=2.814)
    asym = ewma_chart(np.zeros(200), lam=0.1, L=2.814, exact=False)
    assert np.all(np.diff(res.upper) >= 0)
    assert res.upper[-1] == pytest.approx(asym.upper[-1], rel=1e-6)
    assert res.upper[0] == pytest.approx(asym.upper[0] * np.sqrt(1 - 0.9**2))


def test_lambda_one_is_a_shewhart_chart():
    x = np.array([0.5, 3.5, -1.0])
    res = ewma_chart(x, lam=1.0, L=3.0)
    assert np.allclose(res.statistic, x)
    assert res.first_signal == 2


def test_a_sustained_shift_is_detected():
    rng = np.random.default_rng(0)
    early, delays = 0, []
    for _ in range(200):
        x = np.r_[rng.standard_normal(50), rng.standard_normal(100) + 1.0]
        first = ewma_chart(x).first_signal
        if first is not None and first <= 50:
            early += 1
        elif first is not None:
            delays.append(first - 50)
    # An in-control run length of about 500 gives few alarms in 50 batches.
    assert early / 200 < 0.15
    assert np.median(delays) < 12


def test_invalid_lambda():
    with pytest.raises(ValueError):
        ewma_chart([0.0], lam=0.0)


def test_cusum_recursion():
    x = np.array([1.0, 1.0, -2.0, 0.2])
    res = cusum_chart(x, k=0.5, h=1.2)
    assert np.allclose(res.plus, [0.5, 1.0, 0.0, 0.0])
    assert np.allclose(res.minus, [0.0, 0.0, 1.5, 0.8])
    assert res.first_signal == 3 and res.statistic[2] == -1.5


def test_siegmund_matches_published_run_lengths():
    # Two-sided CUSUM with k = 0.5: h = 4 gives about 168, h = 5 about 465
    # (Montgomery, Introduction to Statistical Quality Control, table 9.3).
    assert siegmund_arl(4.0, 0.5) == pytest.approx(168, rel=0.03)
    assert siegmund_arl(5.0, 0.5) == pytest.approx(465, rel=0.03)
    assert siegmund_arl(5.0, 0.5, shift=1.0) == pytest.approx(10.4, rel=0.05)


def test_interval_for_target_run_length():
    h = cusum_interval(0.5, 370.0)
    assert 4.5 < h < 4.9
    assert siegmund_arl(h, 0.5) == pytest.approx(370.0)


def test_hwma_weights_and_variance():
    x = np.array([1.0, 3.0, -1.0, 2.0])
    res = hwma_chart(x, lam=0.2, L=3.0)
    expected = [0.2 * 1.0, 0.2 * 3.0 + 0.8 * 1.0, 0.2 * -1.0 + 0.8 * 2.0, 0.4 + 0.8]
    assert np.allclose(res.statistic, expected)
    rng = np.random.default_rng(0)
    sims = np.array(
        [hwma_chart(rng.standard_normal(6)).statistic for _ in range(20000)]
    )
    sd = np.sqrt(0.1**2 + 0.9**2 / np.r_[np.inf, np.arange(1, 6)])
    assert np.allclose(sims.std(axis=0), sd, rtol=0.03)
    assert np.allclose(hwma_chart(np.zeros(6), L=2.8).upper, 2.8 * sd)


def test_mhwma_reduces_to_squared_hwma_in_one_dimension():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(30)
    uni = hwma_chart(x, lam=0.15)
    multi = mhwma_chart(x[:, None], lam=0.15)
    assert np.allclose(multi.statistic, (uni.statistic / (uni.upper / 2.8)) ** 2)


def test_mhwma_is_chi_square_in_control():
    rng = np.random.default_rng(2)
    cov = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 1.0]])
    chol = np.linalg.cholesky(cov)
    stats = np.array(
        [
            mhwma_chart(rng.standard_normal((10, 3)) @ chol.T, cov=cov).statistic
            for _ in range(5000)
        ]
    )
    # Each statistic is a squared Mahalanobis norm of a normal vector.
    assert np.allclose(stats.mean(axis=0), 3.0, rtol=0.06)
