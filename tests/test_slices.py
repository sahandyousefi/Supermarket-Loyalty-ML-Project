import numpy as np
import pandas as pd

from loyalty_spend.robustness.slices import find_slices


def test_planted_interaction_slice_is_found():
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame(
        {
            "a": rng.choice(["x", "y"], n),
            "b": rng.choice(["u", "v", "w"], n),
            "num": rng.normal(size=n),
        }
    )
    y = rng.normal(size=n)
    noise = np.where((frame["a"] == "y") & (frame["b"] == "w"), 4.0, 1.0)
    pred = y + rng.normal(size=n) * noise
    table = find_slices(y, pred, frame, ["a", "b", "num"])
    flagged = table[table["flagged"]]
    assert flagged.iloc[0]["slice"] == "a | b = y & w"
    assert not table.loc[table["slice"].str.startswith("num ="), "flagged"].any()


def test_no_flags_without_a_problem():
    rng = np.random.default_rng(1)
    n = 2000
    frame = pd.DataFrame({"a": rng.choice(["x", "y", "z"], n)})
    y = rng.normal(size=n)
    table = find_slices(y, y + rng.normal(size=n), frame, ["a"])
    assert not table["flagged"].any()
