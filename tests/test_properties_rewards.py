import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loyalty_spend.business.rewards import RewardSchedule

spend = st.floats(min_value=0.0, max_value=1e5, allow_nan=False)


@st.composite
def schedules(draw, increasing_rates: bool = False):
    n = draw(st.integers(min_value=1, max_value=6))
    thresholds = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    rates = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    if increasing_rates:
        rates = sorted(rates)
    return RewardSchedule(tuple(zip(thresholds, rates)))


@settings(max_examples=200, deadline=None)
@given(schedules(), st.lists(spend, min_size=1, max_size=50))
def test_reward_is_rate_times_spend_and_bounded(schedule, values):
    s = np.array(values)
    rates = {r for _, r in schedule.tiers} | {0.0}
    rate = schedule.rate(s)
    assert set(np.unique(rate)) <= rates
    reward = schedule.reward(s)
    assert np.all(reward >= 0)
    assert np.all(reward <= max(rates) * s + 1e-9)


@settings(max_examples=200, deadline=None)
@given(schedules(), spend, spend)
def test_tier_is_monotone_in_spend(schedule, a, b):
    lo, hi = sorted((a, b))
    assert schedule.tier(lo) <= schedule.tier(hi)


@settings(max_examples=200, deadline=None)
@given(schedules(increasing_rates=True), spend, spend)
def test_reward_is_monotone_when_rates_increase(schedule, a, b):
    lo, hi = sorted((a, b))
    assert schedule.reward(lo) <= schedule.reward(hi) + 1e-9


@given(st.lists(st.floats(0, 100, allow_nan=False), min_size=2, max_size=5))
def test_unsorted_thresholds_are_rejected(thresholds):
    if thresholds == sorted(set(thresholds)):
        return
    with pytest.raises(ValueError):
        RewardSchedule(tuple((t, 0.01) for t in thresholds))
