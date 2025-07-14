import numpy as np
import pytest

from loyalty_spend.business.rewards import (
    MarginalSchedule,
    RewardSchedule,
    forecast_liability,
    gaussian_qmc_sampler,
    gaussian_sampler,
    near_threshold,
    residual_sampler,
    simulate_liability,
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


def test_residual_bootstrap_is_a_special_case_of_the_sampler():
    rng = np.random.default_rng(9)
    pred = rng.uniform(90, 150, size=40)
    resid = rng.normal(scale=5, size=300)
    a = forecast_liability(range(40), pred, resid, SCHEDULE, 700, 3, batch=128)
    b = simulate_liability(
        range(40), pred, residual_sampler(pred, resid), SCHEDULE, 700, 3, batch=128
    )
    np.testing.assert_allclose(a.total_draws, b.total_draws)


def test_customer_specific_uncertainty_changes_the_cost():
    # Same point forecast and the same average noise, but one customer is
    # far less predictable: only the heteroscedastic sampler sees that the
    # uncertain customer may cross the top threshold.
    mean = np.array([130.0, 130.0])
    flat = simulate_liability(
        [1, 2], mean, gaussian_sampler(mean, np.sqrt(50.0)), SCHEDULE, 20000, 0
    )
    uneven = simulate_liability(
        [1, 2], mean, gaussian_sampler(mean, [1.0, np.sqrt(99.0)]), SCHEDULE, 20000, 0
    )
    p_flat = flat.per_customer["tier_change_probability"].to_numpy()
    p_uneven = uneven.per_customer["tier_change_probability"].to_numpy()
    assert p_flat[0] == pytest.approx(p_flat[1], abs=0.02)
    assert p_uneven[0] < 0.01 and p_uneven[1] > 0.3


def test_marginal_schedule_matches_a_hand_computation():
    marginal = MarginalSchedule(SCHEDULE.tiers)
    # 130: 2% of (125 - 100) + 3% of (130 - 125).
    assert marginal.reward(130.0) == pytest.approx(0.02 * 25 + 0.03 * 5)
    assert marginal.reward(90.0) == 0.0
    assert marginal.reward(150.0) == pytest.approx(0.5 + 0.03 * 15 + 0.05 * 10)
    assert marginal.rate(130.0) == pytest.approx(marginal.reward(130.0) / 130)
    np.testing.assert_array_equal(marginal.tier([90, 130]), SCHEDULE.tier([90, 130]))


def test_marginal_reward_is_continuous_where_whole_spend_jumps():
    marginal = MarginalSchedule(SCHEDULE.tiers)
    eps = 1e-6
    for threshold in (100.0, 125.0, 140.0):
        jump_whole = SCHEDULE.reward(threshold) - SCHEDULE.reward(threshold - eps)
        jump_marginal = marginal.reward(threshold) - marginal.reward(threshold - eps)
        assert jump_whole > 1.0
        assert abs(jump_marginal) < 1e-6


def test_marginal_schedule_works_in_the_simulation():
    marginal = MarginalSchedule(SCHEDULE.tiers)
    fc = forecast_liability(
        [1, 2], [126.0, 141.0], np.linspace(-5, 5, 50), marginal, 500, 0
    )
    assert fc.summary()["expected_total"] > 0


def test_cost_to_spend_ratio_and_tier_shares():
    fc = forecast_liability(
        range(3), [90.0, 130.0, 150.0], np.zeros(5), SCHEDULE, 50, 0
    )
    expected_cost = 0.0 + 0.03 * 130 + 0.05 * 150
    assert fc.cost_to_spend() == pytest.approx(expected_cost / 370.0)
    lo, hi = fc.ratio_interval()
    assert lo == pytest.approx(hi)
    # Shares: below the first threshold, then tiers from 0 upwards. The
    # first threshold is 0, so nobody is below it.
    np.testing.assert_allclose(fc.tier_shares, [0.0, 1 / 3, 0.0, 1 / 3, 1 / 3])


def test_monte_carlo_errors_shrink_with_more_draws():
    rng = np.random.default_rng(3)
    pred = rng.uniform(95, 150, size=100)
    resid = rng.normal(scale=6, size=400)
    small = forecast_liability(
        range(100), pred, resid, SCHEDULE, 400, 0
    ).standard_errors()
    large = forecast_liability(
        range(100), pred, resid, SCHEDULE, 6400, 0
    ).standard_errors()
    for key in ("expected_total", "p05", "p50", "p95"):
        assert large[key] < small[key]
    assert large["expected_total"] == pytest.approx(
        small["expected_total"] / 4, rel=0.3
    )


def test_quasi_monte_carlo_estimates_vary_less_than_monte_carlo():
    import warnings

    rng = np.random.default_rng(4)
    mean = rng.uniform(95, 150, size=50)
    mc, qmc = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol warns when m is not a power of two
        for rep in range(12):
            mc.append(
                simulate_liability(
                    range(50),
                    mean,
                    gaussian_sampler(mean, 8.0),
                    SCHEDULE,
                    256,
                    rep,
                    batch=256,
                ).total_draws.mean()
            )
            qmc.append(
                simulate_liability(
                    range(50),
                    mean,
                    gaussian_qmc_sampler(mean, 8.0, seed=rep),
                    SCHEDULE,
                    256,
                    0,
                    batch=256,
                ).total_draws.mean()
            )
    assert np.std(qmc) < np.std(mc)
    assert np.mean(qmc) == pytest.approx(np.mean(mc), rel=0.01)
