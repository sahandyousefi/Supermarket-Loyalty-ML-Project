import numpy as np
import pytest

from loyalty_spend.causal.uplift import dr_gain_curve, qini_coefficient, qini_curve


@pytest.fixture(scope="module")
def experiment():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.uniform(size=n)
    tau = 4.0 * x
    t = rng.integers(0, 2, size=n)
    y = 100 + 10 * x + tau * t + rng.normal(scale=2, size=n)
    return x, tau, t, y


def test_true_ranking_beats_random_and_reverse(experiment):
    x, tau, t, y = experiment
    good = qini_coefficient(qini_curve(tau, t, y), len(y))
    flat = qini_coefficient(qini_curve(np.zeros_like(tau), t, y), len(y))
    bad = qini_coefficient(qini_curve(-tau, t, y), len(y))
    assert good > 0 > bad
    assert abs(flat) < 0.25 * good


def test_curve_ends_at_the_total_effect(experiment):
    _, tau, t, y = experiment
    curve = qini_curve(tau, t, y)
    total = y[t == 1].sum() - y[t == 0].sum() * t.sum() / (1 - t).sum()
    assert curve["gain"].iloc[-1] == pytest.approx(total)
    assert curve["fraction"].iloc[-1] == pytest.approx(1.0)


def test_dr_gain_with_oracle_scores(experiment):
    _, tau, _, _ = experiment
    curve = dr_gain_curve(tau, tau)
    # With the true effects as scores, the gain of the top half is the sum of
    # the largest half of the effects.
    half = curve.iloc[(curve["fraction"] - 0.5).abs().argmin()]
    assert half["gain"] == pytest.approx(
        np.sort(tau)[::-1][: int(0.5 * len(tau))].sum(), rel=0.02
    )
    assert qini_coefficient(curve, len(tau)) > 0
