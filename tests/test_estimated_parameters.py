import numpy as np
import pytest

from loyalty_spend.monitoring.estimated_parameters import (
    conditional_arl,
    estimation_effect,
    ewma_arl_markov,
)
from loyalty_spend.monitoring.run_length import simulate_run_lengths


def test_markov_chain_matches_lucas_and_saccucci():
    assert ewma_arl_markov(0.1, 2.814) == pytest.approx(500, rel=0.01)
    assert ewma_arl_markov(0.1, 2.814, shift=1.0) == pytest.approx(10.3, rel=0.03)


def test_markov_chain_agrees_with_simulation_under_scale_change():
    exact = ewma_arl_markov(0.2, 2.86, shift=0.3, scale=1.2)
    rl = simulate_run_lengths(
        "ewma",
        {"lam": 0.2, "L": 2.86, "exact": False},
        3000,
        3000,
        shift=0.3,
        scale=1.2,
        seed=1,
    )
    assert rl.mean() == pytest.approx(exact, abs=4 * rl.std() / np.sqrt(len(rl)))


def test_estimation_spreads_the_conditional_run_length():
    small = conditional_arl(25, n_phase1=300)
    large = conditional_arl(1000, n_phase1=300)
    assert np.std(small) > 3 * np.std(large)
    table = estimation_effect(sizes=(25, 1000), n_phase1=300)
    assert table["share_below_80pct"].iloc[0] > table["share_below_80pct"].iloc[1]
