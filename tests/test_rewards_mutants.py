"""Checks added for reward-module behaviour that mutation testing left open."""

import numpy as np
import pandas as pd
import pytest

from loyalty_spend.business.rewards import (
    LiabilityForecast,
    MarginalSchedule,
    RewardSchedule,
    batch_size_for,
    near_threshold,
    residual_sampler,
    simulate_liability,
)


def _forecast(draws, spend=None):
    pc = pd.DataFrame({"plug_in_reward": [0.0]})
    return LiabilityForecast(pc, np.asarray(draws, float), spend)


def test_rate_in_the_first_tier_and_below_it():
    schedule = RewardSchedule(((10.0, 0.005), (100.0, 0.01)))
    assert np.allclose(schedule.rate([5.0, 10.0, 50.0, 100.0]), [0, 0.005, 0.005, 0.01])


def test_tail_risk_values_on_a_discrete_distribution():
    risk = _forecast([1.0, 2.0, 3.0, 4.0, 10.0]).tail_risk(0.8)
    # The 80% quantile interpolates to 5.2; only the draw of 10 lies above it.
    assert risk["var"] == pytest.approx(5.2)
    assert risk["expected_shortfall"] == pytest.approx(10.0)
    assert risk["var_over_expected"] == pytest.approx(5.2 - 4.0)
    assert risk["shortfall_over_expected"] == pytest.approx(10.0 - 4.0)
    # A draw exactly at the quantile belongs to the tail.
    assert _forecast([1.0, 1.0, 1.0, 5.0]).tail_risk(0.5)["expected_shortfall"] == 2.0
    for level in (0.0, 1.0):
        with pytest.raises(ValueError):
            _forecast([1.0, 2.0]).tail_risk(level)


def test_standard_errors_follow_their_definitions():
    draws = np.arange(40, dtype=float)
    se = _forecast(draws).standard_errors(levels=(0.5,), n_batches=4)
    assert se["expected_total"] == pytest.approx(draws.std(ddof=1) / np.sqrt(40))
    medians = np.array([np.median(b) for b in np.array_split(draws, 4)])
    assert se["p50"] == pytest.approx(medians.std(ddof=1) / 2)


def test_ratio_interval_is_central():
    cost = np.arange(1.0, 101.0)
    spend = np.full(100, 10.0)
    lo, hi = _forecast(cost, spend).ratio_interval(0.5)
    assert lo == pytest.approx(np.quantile(cost / 10, 0.25))
    assert hi == pytest.approx(np.quantile(cost / 10, 0.75))


def test_residual_sampler_adds_the_residual():
    draws = residual_sampler([100.0, 50.0], [2.0])(np.random.default_rng(0), 3)
    assert np.allclose(draws, [[102.0, 52.0]] * 3)


def test_simulation_clips_negative_spend_and_averages_rewards():
    schedule = RewardSchedule(((0.0, 0.1), (1.0, 0.2)))

    def sampler(rng, m):
        return np.tile([0.5, -3.0, 2.0], (m, 1))

    out = simulate_liability(np.arange(3), [0.5, 0.0, 2.0], sampler, schedule, 7)
    assert np.allclose(out.per_customer["expected_reward"], [0.05, 0.0, 0.4])
    assert np.allclose(out.total_draws, 0.45)
    assert np.allclose(out.tier_shares, [0.0, 2 / 3, 1 / 3])


def test_near_threshold_includes_the_boundary():
    pc = pd.DataFrame({"tier_change_probability": [0.2, 0.19, 0.5]})
    forecast = LiabilityForecast(pc, np.zeros(1))
    assert list(near_threshold(forecast, 0.2)["tier_change_probability"]) == [0.5, 0.2]


def test_single_customer_batch_size():
    assert batch_size_for(1, 1.0) == 2**20 // 48
    assert batch_size_for(0, 1.0) == 2**20 // 48


def test_marginal_rate_for_small_and_zero_spend():
    schedule = MarginalSchedule(((0.0, 0.1), (1.0, 0.2)))
    assert np.allclose(schedule.rate([0.0, 0.5, 2.0]), [0.0, 0.1, 0.15])
    assert np.allclose(schedule.reward([-1.0, 0.5]), [0.0, 0.05])


def test_tier_shares_when_the_top_tier_is_never_reached():
    schedule = RewardSchedule(((0.0, 0.0), (100.0, 0.01), (200.0, 0.02)))

    def sampler(rng, m):
        return np.full((m, 2), 50.0)

    out = simulate_liability(np.arange(2), [50.0, 50.0], sampler, schedule, 5)
    assert np.allclose(out.tier_shares, [0.0, 1.0, 0.0, 0.0])
