import numpy as np
import pandas as pd
import pytest

from loyalty_spend.robustness.subgroups import subgroup_errors


def test_noisy_group_stands_out():
    rng = np.random.default_rng(0)
    g = np.repeat(["a", "b"], 500)
    y = rng.normal(size=1000)
    pred = (
        y
        + np.where(
            g == "b", rng.normal(scale=3, size=1000), rng.normal(scale=1, size=1000)
        )
        + np.where(g == "a", 0.5, 0.0)
    )
    table = subgroup_errors(
        y, pred, pd.DataFrame({"g": g}), ["g"], n_boot=300
    ).set_index("level")
    assert table.loc["b", "rmse_lower"] > table.loc["a", "rmse_upper"]
    assert table.loc["a", "bias_lower"] > 0
    assert table.loc["b", "bias_lower"] < 0 < table.loc["b", "bias_upper"]
    assert table.loc["b", "rmse_ratio"] > 1


def test_small_groups_are_skipped():
    frame = pd.DataFrame({"g": ["x"] * 5 + ["y"] * 50})
    table = subgroup_errors(np.zeros(55), np.ones(55), frame, ["g"], min_size=10)
    assert list(table["level"]) == ["y"]
    assert table["rmse"].iloc[0] == pytest.approx(1.0)
