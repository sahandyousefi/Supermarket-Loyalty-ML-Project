import numpy as np
import pytest

from loyalty_spend.business.rewards import MarginalSchedule, RewardSchedule
from loyalty_spend.business.sorted_cost import SortedDraws

TIERS = ((0.0, 0.0), (100.0, 0.01), (120.0, 0.02), (140.0, 0.03))


@pytest.fixture(scope="module")
def spend():
    rng = np.random.default_rng(0)
    draws = rng.normal(125, 25, size=(300, 80))
    # Values exactly on a threshold must fall into the higher tier.
    draws[:, :5] = [100.0, 120.0, 140.0, 0.0, -3.0]
    return draws


@pytest.mark.parametrize("cls", [RewardSchedule, MarginalSchedule])
def test_totals_match_direct_application(spend, cls):
    schedule = cls(TIERS)
    direct = schedule.reward(np.clip(spend, 0, None)).sum(axis=1)
    fast = SortedDraws(spend).totals(schedule)
    assert np.allclose(fast, direct, rtol=0, atol=1e-9)


def test_negative_first_threshold_in_marginal_schedule(spend):
    schedule = MarginalSchedule(((-10.0, 0.01), (130.0, 0.02)))
    direct = schedule.reward(np.clip(spend, 0, None)).sum(axis=1)
    assert np.allclose(SortedDraws(spend).totals(schedule), direct)


def test_reach_matches_tier_shares(spend):
    draws = SortedDraws(spend)
    schedule = RewardSchedule(TIERS)
    direct = (schedule.tier(np.clip(spend, 0, None)) >= 1).mean()
    assert draws.reach(100.0) == pytest.approx(direct)
    assert draws.reach(0.0) == 1.0
    assert draws.reach(1e9) == 0.0


def test_below_counts_and_sums(spend):
    draws = SortedDraws(spend)
    count, total = draws.below(120.0)
    clipped = np.clip(spend, 0, None)
    mask = clipped < 120.0
    assert np.array_equal(count, mask.sum(axis=1))
    assert np.allclose(total, np.where(mask, clipped, 0).sum(axis=1))


def test_threshold_search_is_unchanged_by_sorted_costing():
    from loyalty_spend.business.design import search_thresholds
    from loyalty_spend.business.rewards import gaussian_sampler

    rng = np.random.default_rng(3)
    mean = rng.normal(125, 20, 60)
    sampler = gaussian_sampler(mean, np.full(60, 3.0))
    schedule = RewardSchedule(TIERS)
    table = search_thresholds(sampler, schedule, budget=150.0, n_simulations=400)
    spend = np.clip(sampler(np.random.default_rng(0), 400), 0, None)
    for _, row in table.head(5).iterrows():
        cuts = [float(x) for x in row["thresholds"].split(",")]
        cand = RewardSchedule(((0.0, 0.0), *zip(cuts, (0.01, 0.02, 0.03), strict=True)))
        cost = cand.reward(spend).sum(axis=1)
        assert row["expected_cost"] == pytest.approx(cost.mean())
        assert row["reach"] == pytest.approx((cand.tier(spend) >= 1).mean())
