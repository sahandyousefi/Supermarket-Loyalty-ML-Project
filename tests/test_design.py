import numpy as np
import pytest

from loyalty_spend.business.design import (
    compare_schedules,
    cost_sensitivity,
    independent_difference_se,
    schedule_cost_draws,
)
from loyalty_spend.business.rewards import (
    MarginalSchedule,
    RewardSchedule,
    gaussian_sampler,
)

TIERS = ((0.0, 0.0), (100.0, 0.02), (125.0, 0.03), (140.0, 0.05))


@pytest.fixture(scope="module")
def sampler():
    mean = np.random.default_rng(0).uniform(95, 150, size=400)
    return gaussian_sampler(mean, 6.0)


def test_common_random_numbers_shrink_the_difference_error(sampler):
    # Two similar schedules: moving the top threshold from 140 to 142.
    moved = (*TIERS[:3], (142.0, 0.05))
    schedules = {"current": RewardSchedule(TIERS), "moved": RewardSchedule(moved)}
    table = compare_schedules(sampler, schedules, "current", n_simulations=2000)
    table = table.set_index("schedule")
    assert table.loc["current", "difference"] == 0.0
    assert table.loc["moved", "difference"] < 0
    a = schedule_cost_draws(sampler, {"x": schedules["current"]}, 2000, seed=1)["x"]
    b = schedule_cost_draws(sampler, {"x": schedules["moved"]}, 2000, seed=2)["x"]
    assert table.loc["moved", "difference_se"] < 0.5 * independent_difference_se(a, b)


def test_marginal_schedule_costs_less_with_the_same_rates(sampler):
    schedules = {"whole": RewardSchedule(TIERS), "marginal": MarginalSchedule(TIERS)}
    table = compare_schedules(sampler, schedules, "whole", n_simulations=500)
    assert table.set_index("schedule").loc["marginal", "difference"] < 0


def test_unknown_reference(sampler):
    with pytest.raises(KeyError):
        compare_schedules(sampler, {"a": RewardSchedule(TIERS)}, "b", n_simulations=10)


def test_rate_sensitivity_is_the_spend_in_the_tier():
    # Deterministic spend: d cost / d rate of a whole-spend tier is the total
    # spend of the members in that tier.
    spend = np.array([90.0, 130.0, 150.0, 160.0])
    table = cost_sensitivity(
        gaussian_sampler(spend, 1e-9), RewardSchedule(TIERS), n_simulations=20
    )
    by_tier = table.set_index("tier")
    assert by_tier.loc[2, "d_cost_d_rate"] == pytest.approx(130.0, rel=1e-6)
    assert by_tier.loc[3, "d_cost_d_rate"] == pytest.approx(310.0, rel=1e-6)


def test_raising_a_crowded_threshold_lowers_cost(sampler):
    table = cost_sensitivity(sampler, RewardSchedule(TIERS), n_simulations=1000)
    by_tier = table.set_index("tier")
    assert (by_tier.loc[[1, 2, 3], "d_cost_d_threshold"] < 0).all()
    assert np.isnan(by_tier.loc[0, "d_cost_d_threshold"])
