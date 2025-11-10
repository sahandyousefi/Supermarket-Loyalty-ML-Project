import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from loyalty_spend.business.rewards import MarginalSchedule, RewardSchedule
from loyalty_spend.business.sorted_cost import SortedDraws

# Integer-valued spend makes ties with integer thresholds common.
spend_values = st.integers(min_value=-5, max_value=200).map(float)
spend_matrices = st.tuples(st.integers(1, 6), st.integers(1, 12)).flatmap(
    lambda shape: arrays(np.float64, shape, elements=spend_values)
)


@st.composite
def schedules(draw):
    k = draw(st.integers(1, 4))
    cuts = sorted(draw(st.sets(st.integers(1, 190), min_size=k, max_size=k)))
    rates = draw(st.lists(st.floats(0, 0.1), min_size=k, max_size=k))
    first = draw(st.floats(0, 0.05))
    return ((0.0, first), *zip(map(float, cuts), rates, strict=True))


@settings(max_examples=200, deadline=None)
@given(spend=spend_matrices, tiers=schedules(), marginal=st.booleans())
def test_sorted_totals_equal_direct_application(spend, tiers, marginal):
    schedule = (MarginalSchedule if marginal else RewardSchedule)(tiers)
    direct = schedule.reward(np.clip(spend, 0, None)).sum(axis=1)
    assert np.allclose(SortedDraws(spend).totals(schedule), direct, atol=1e-9)


@settings(max_examples=200, deadline=None)
@given(spend=spend_matrices, threshold=st.floats(-10, 250))
def test_reach_is_the_share_at_or_above_the_threshold(spend, threshold):
    expected = (np.clip(spend, 0, None) >= threshold).mean()
    assert np.isclose(SortedDraws(spend).reach(threshold), expected)


@settings(max_examples=100, deadline=None)
@given(spend=spend_matrices, tiers=schedules())
def test_totals_do_not_depend_on_member_order(spend, tiers):
    schedule = RewardSchedule(tiers)
    shuffled = np.random.default_rng(0).permuted(spend, axis=1)
    assert np.allclose(
        SortedDraws(spend).totals(schedule), SortedDraws(shuffled).totals(schedule)
    )
