import numpy as np
import pytest

from loyalty_spend.causal.policy import compare_policies, policy_value, threshold_rule


@pytest.fixture(scope="module")
def scores():
    rng = np.random.default_rng(0)
    tau = rng.uniform(-2, 6, size=20000)
    psi = tau + rng.normal(scale=10, size=20000)
    return tau, psi


def test_threshold_rule():
    np.testing.assert_array_equal(threshold_rule([1.0, 3.0], 2.0), [False, True])
    np.testing.assert_array_equal(
        threshold_rule([1.0, 3.0], 2.0, margin=0.5), [False, False]
    )
    with pytest.raises(ValueError):
        threshold_rule([1.0], 0.0, margin=0.0)


def test_policy_value_estimates_the_true_value(scores):
    tau, psi = scores
    rule = threshold_rule(tau, cost=2.0)
    out = policy_value(rule, psi, cost=2.0)
    truth = np.mean(rule * (tau - 2.0))
    assert out["ci_lower"] < truth < out["ci_upper"]
    assert out["share_treated"] == pytest.approx(0.5, abs=0.02)


def test_targeting_beats_treating_everybody_when_some_lose(scores):
    tau, psi = scores
    table = compare_policies(tau, psi, cost=2.0).set_index("policy")
    assert table.loc["nobody", "value"] == 0.0
    assert (
        table.loc["predicted profit above cost", "value"]
        > table.loc["everybody", "value"]
    )
