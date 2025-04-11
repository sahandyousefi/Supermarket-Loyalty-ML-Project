import numpy as np
import pytest

from loyalty_spend.business.rewards import (
    RewardSchedule,
    forecast_liability,
    near_threshold,
)

SCHEDULE = RewardSchedule(((0.0, 0.0), (100.0, 0.02), (125.0, 0.03), (140.0, 0.05)))


def test_reward_rates_by_tier():
    spend = np.array([50.0, 100.0, 124.99, 125.0, 150.0])
    assert np.allclose(SCHEDULE.rate(spend), [0.0, 0.02, 0.02, 0.03, 0.05])
    assert SCHEDULE.reward(150.0) == pytest.approx(7.5)


@pytest.mark.parametrize(
    "tiers", [((0.0, 0.0), (0.0, 0.1)), ((10.0, 0.1), (5.0, 0.2)), ((0.0, 1.5),)]
)
def test_invalid_schedules(tiers):
    with pytest.raises(ValueError):
        RewardSchedule(tiers)


def test_without_noise_forecast_equals_plug_in():
    fc = forecast_liability([1, 2], [110.0, 150.0], np.zeros(10), SCHEDULE, 200, 0)
    s = fc.summary()
    assert s["expected_total"] == pytest.approx(s["plug_in_total"])
    assert (fc.per_customer["tier_change_probability"] == 0).all()


def test_jensen_gap_near_threshold():
    resid = np.random.default_rng(0).normal(0, 3, 1000)
    fc = forecast_liability([1], [139.0], resid, SCHEDULE, 4000, 0)
    row = fc.per_customer.iloc[0]
    assert row["expected_reward"] > row["plug_in_reward"]
    assert 0.2 < row["tier_change_probability"] < 0.5
    assert len(near_threshold(fc)) == 1


def test_summary_quantiles_are_ordered():
    resid = np.random.default_rng(1).normal(0, 5, 500)
    fc = forecast_liability(range(50), np.full(50, 130.0), resid, SCHEDULE, 1000, 0)
    s = fc.summary()
    assert s["p05"] <= s["p50"] <= s["p95"]
