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
