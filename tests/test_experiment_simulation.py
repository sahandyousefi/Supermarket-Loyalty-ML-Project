import pytest

from loyalty_spend.experiment.sequential import boundaries
from loyalty_spend.experiment.simulation import simulate_trial, summarise_trials

LOOKS = [0.5, 1.0]


@pytest.fixture(scope="module")
def bounds():
    return boundaries(LOOKS)


def test_type_one_error_is_controlled(train_df, bounds):
    trials = simulate_trial(train_df, 200, 0.0, LOOKS, bounds, n_reps=400, seed=1)
    s = summarise_trials(trials, 0.0)
    assert s["rejection_rate"] < 0.05 + 3 * s["rejection_se"]
    assert abs(s["bias"]) < 0.05


def test_adjustment_detects_a_small_effect_that_raw_means_miss(train_df, bounds):
    lin = simulate_trial(train_df, 200, 0.3, LOOKS, bounds, "lin", 200, seed=2)
    raw = simulate_trial(train_df, 200, 0.3, LOOKS, bounds, "difference", 200, seed=2)
    assert summarise_trials(lin, 0.3)["rejection_rate"] > 0.9
    assert summarise_trials(raw, 0.3)["rejection_rate"] < 0.2
    assert set(lin["stopped_at"]) <= {1, 2}
