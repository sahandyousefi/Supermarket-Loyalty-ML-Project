import numpy as np
import pandas as pd
import pytest
from scipy import stats

from loyalty_spend.analysis.drivers import games_howell, tukey_table


def _frame(rng, sizes, means, sds):
    groups = np.repeat([f"g{i}" for i in range(len(sizes))], sizes)
    values = np.concatenate(
        [rng.normal(m, s, n) for n, m, s in zip(sizes, means, sds, strict=True)]
    )
    return pd.DataFrame({"g": groups, "spend": values})


def test_close_to_tukey_with_equal_variances_and_sizes():
    df = _frame(np.random.default_rng(0), [200] * 3, [0.0, 0.2, 0.5], [1.0] * 3)
    gh, tk = games_howell(df, "g"), tukey_table(df, "g")
    assert np.allclose(gh["diff"], tk["diff"])
    assert np.allclose(gh["p_value"], tk["p_value"], atol=0.01)
    assert np.allclose(gh["ci_low"], tk["ci_low"], atol=0.02)


def test_interval_and_p_value_agree():
    df = _frame(np.random.default_rng(1), [30, 80, 15], [0.0, 0.6, 0.1], [0.5, 2, 4])
    gh = games_howell(df, "g")
    excludes_zero = (gh["ci_low"] > 0) | (gh["ci_high"] < 0)
    assert (excludes_zero == (gh["p_value"] < 0.05)).all()


def test_familywise_error_with_unequal_variances():
    rng = np.random.default_rng(2)
    gh_hits = tk_hits = 0
    reps = 200
    for _ in range(reps):
        # A small group with a large spread inflates Tukey's error.
        df = _frame(rng, [10, 60, 60], [0, 0, 0], [4.0, 1.0, 1.0])
        gh_hits += (games_howell(df, "g", intervals=False)["p_value"] < 0.05).any()
        groups = [part["spend"].to_numpy() for _, part in df.groupby("g")]
        pvalues = stats.tukey_hsd(*groups).pvalue
        tk_hits += (pvalues[np.triu_indices(3, 1)] < 0.05).any()
    assert gh_hits / reps == pytest.approx(0.05, abs=0.04)
    assert tk_hits / reps > 2 * gh_hits / reps
