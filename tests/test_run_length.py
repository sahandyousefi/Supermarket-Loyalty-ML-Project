import numpy as np
import pytest

from loyalty_spend.monitoring.control_charts import (
    cusum_chart,
    ewma_chart,
    hwma_chart,
    mhwma_chart,
)
from loyalty_spend.monitoring.run_length import simulate_run_lengths, summarise


def test_ewma_in_control_run_length_matches_lucas_and_saccucci():
    rl = simulate_run_lengths(
        "ewma", {"lam": 0.1, "L": 2.814, "exact": False}, n_reps=2000, seed=0
    )
    s = summarise(rl, 5000)
    assert s.censored == 0
    assert s.mean == pytest.approx(500, abs=4 * s.se)


def test_cusum_run_lengths_match_published_values():
    ic = summarise(simulate_run_lengths("cusum", {"k": 0.5, "h": 5.0}, 2000), 5000)
    assert ic.mean == pytest.approx(465, abs=4 * ic.se)
    oc = simulate_run_lengths("cusum", {"k": 0.5, "h": 5.0}, 2000, shift=1.0)
    assert oc.mean() == pytest.approx(10.4, rel=0.05)


@pytest.mark.parametrize(
    ("chart", "params", "single"),
    [
        ("ewma", {"lam": 0.2, "L": 2.9}, lambda x: ewma_chart(x, 0.2, 2.9)),
        ("cusum", {"k": 0.5, "h": 4.0}, lambda x: cusum_chart(x, 0.5, 4.0)),
        ("hwma", {"lam": 0.1, "L": 2.8}, lambda x: hwma_chart(x, 0.1, 2.8)),
    ],
)
def test_vectorised_signals_agree_with_the_charts(chart, params, single):
    rl = simulate_run_lengths(chart, params, n_reps=50, horizon=300, shift=0.5, seed=3)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 300)) + 0.5
    expected = [single(row).first_signal or 301 for row in x]
    assert np.array_equal(rl, expected)


def test_mhwma_vectorised_agrees():
    rl = simulate_run_lengths(
        "mhwma", {"lam": 0.1, "h": 12.0}, 30, 200, shift=0.4, dim=3, seed=4
    )
    rng = np.random.default_rng(4)
    x = rng.standard_normal((30, 200, 3)) + 0.4
    expected = [mhwma_chart(row, 0.1, 12.0).first_signal or 201 for row in x]
    assert np.array_equal(rl, expected)


def test_steady_state_runs_drop_false_alarms():
    rl = simulate_run_lengths("ewma", {"lam": 0.1, "L": 2.0}, 300, 200, tau=100)
    assert (rl >= 1).all() and len(rl) < 300
