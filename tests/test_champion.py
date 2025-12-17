import numpy as np
import pytest

from loyalty_spend.ops.champion import compare_predictions


@pytest.fixture(scope="module")
def outcome():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, size=2000)
    return rng, y


def test_better_challenger_wins(outcome):
    rng, y = outcome
    old = y + rng.normal(scale=3.0, size=len(y))
    new = y + rng.normal(scale=2.0, size=len(y))
    result = compare_predictions(y, old, new, n_boot=500)
    assert result.challenger_rmse < result.champion_rmse
    assert result.challenger_wins
    assert result.p_value < 1e-6


def test_equal_models_do_not_trigger_a_switch(outcome):
    rng, y = outcome
    old = y + rng.normal(scale=2.0, size=len(y))
    new = y + rng.normal(scale=2.0, size=len(y))
    result = compare_predictions(y, old, new, margin=0.05, n_boot=500)
    assert not result.challenger_wins
    assert result.ci_lower < 0.05


def test_identical_predictions(outcome):
    _, y = outcome
    result = compare_predictions(y, y + 1.0, y + 1.0, n_boot=100)
    assert result.rmse_difference == 0.0 and result.p_value == 1.0


def test_interval_comparison_and_promotion_rule():
    import numpy as np

    from loyalty_spend.ops.champion import (
        compare_intervals,
        compare_predictions,
        promotion_decision,
    )

    rng = np.random.default_rng(0)
    n = 2000
    noisy = rng.random(n) < 0.3
    sd = np.where(noisy, 3.0, 0.5)
    y = rng.normal(0, sd)
    point = np.zeros(n)
    pooled = 1.645 * np.sqrt(np.mean(sd**2))
    champion = (point - pooled, point + pooled)
    challenger = (point - 1.645 * sd, point + 1.645 * sd)
    iv = compare_intervals(y, champion, challenger)
    assert iv.score_difference < 0 and iv.ci_upper < 0
    same = compare_predictions(y, point, point)
    decision = promotion_decision(same, iv)
    assert decision.promote and "interval" in decision.reason
    worse = compare_predictions(y, point, point + 0.5)
    assert not promotion_decision(worse, iv).promote
