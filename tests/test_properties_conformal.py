import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from loyalty_spend.uncertainty.conformal import conformal_quantile

scores = st.lists(
    st.floats(min_value=0, max_value=1e4, allow_nan=False), min_size=1, max_size=300
)
alphas = st.floats(min_value=0.01, max_value=0.5)


@settings(max_examples=200, deadline=None)
@given(scores, alphas)
def test_quantile_is_a_calibration_score_or_infinite(values, alpha):
    q = conformal_quantile(np.array(values), alpha)
    assert math.isinf(q) or q in values


@settings(max_examples=200, deadline=None)
@given(scores, alphas)
def test_enough_scores_lie_below_the_quantile(values, alpha):
    n = len(values)
    q = conformal_quantile(np.array(values), alpha)
    assume(not math.isinf(q))
    assert np.sum(np.array(values) <= q) >= math.ceil((n + 1) * (1 - alpha))


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(0, 1e4, allow_nan=False), min_size=1, max_size=300, unique=True),
    alphas,
)
def test_rank_coverage_is_within_the_finite_sample_bounds(values, alpha):
    # With distinct exchangeable scores, a new score falls below the k-th
    # smallest calibration score with probability k / (n + 1).
    n = len(values)
    q = conformal_quantile(np.array(values), alpha)
    assume(not math.isinf(q))
    k = int(np.sum(np.array(values) <= q))
    assert 1 - alpha - 1e-12 <= k / (n + 1) < 1 - alpha + 1 / (n + 1) + 1e-12


@settings(max_examples=200, deadline=None)
@given(scores, alphas, alphas)
def test_smaller_alpha_gives_a_wider_quantile(values, a, b):
    lo, hi = sorted((a, b))
    assert conformal_quantile(np.array(values), lo) >= conformal_quantile(
        np.array(values), hi
    )


@settings(max_examples=100, deadline=None)
@given(scores, alphas, st.randoms(use_true_random=False))
def test_quantile_ignores_the_order_of_scores(values, alpha, rnd):
    shuffled = list(values)
    rnd.shuffle(shuffled)
    assert conformal_quantile(np.array(values), alpha) == conformal_quantile(
        np.array(shuffled), alpha
    )
