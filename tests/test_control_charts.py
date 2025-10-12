import numpy as np
import pytest

from loyalty_spend.monitoring.control_charts import ewma_chart


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
