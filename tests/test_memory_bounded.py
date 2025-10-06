import tracemalloc

import numpy as np

from loyalty_spend.business.rewards import (
    RewardSchedule,
    batch_size_for,
    gaussian_sampler,
    residual_sampler,
    simulate_liability,
)

SCHEDULE = RewardSchedule(((0.0, 0.0), (100.0, 0.01), (125.0, 0.02), (140.0, 0.05)))


def test_batch_size_respects_the_budget():
    assert batch_size_for(1_000_000, 48) == 1
    assert batch_size_for(1_000, 48) == 1048
    assert batch_size_for(10**9, 1) == 1


def test_bounded_run_gives_identical_draws():
    rng = np.random.default_rng(0)
    pred = rng.normal(125, 20, 2000)
    for sampler in (
        gaussian_sampler(pred, 2.0),
        residual_sampler(pred, rng.normal(0, 2, 300)),
    ):
        full = simulate_liability(np.arange(2000), pred, sampler, SCHEDULE, 300, 1)
        small = simulate_liability(
            np.arange(2000), pred, sampler, SCHEDULE, 300, 1, max_memory_mb=1
        )
        assert np.array_equal(full.total_draws, small.total_draws)
        # Per-customer means are summed in a different order, so allow rounding.
        assert np.allclose(full.per_customer, small.per_customer, rtol=1e-12)


def test_peak_memory_stays_near_the_budget():
    n = 20_000
    pred = np.random.default_rng(1).normal(125, 20, n)
    sampler = gaussian_sampler(pred, 2.0)

    def peak(**kwargs):
        tracemalloc.start()
        simulate_liability(np.arange(n), pred, sampler, SCHEDULE, 200, 0, **kwargs)
        _, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top / 2**20

    unbounded, bounded = peak(), peak(max_memory_mb=8)
    assert bounded < 12 < unbounded
