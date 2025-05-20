import numpy as np
import pandas as pd
import pytest

from loyalty_spend.data.profile import profile_frame, top_levels


def test_profile_of_a_small_frame():
    df = pd.DataFrame(
        {
            "amount": [1.0, 2.0, np.nan, 4.0, 0.0],
            "text_amount": ["1", "2", None, "3", "4"],
            "level": ["a", "b", "a", None, "a"],
        }
    )
    p = profile_frame(df)
    assert p.loc["amount", "kind"] == "numeric"
    assert p.loc["text_amount", "kind"] == "numeric"
    assert p.loc["level", "kind"] == "category"
    assert p.loc["amount", "missing"] == 1
    assert p.loc["amount", "missing_share"] == pytest.approx(0.2)
    assert p.loc["amount", "zeros"] == 1
    assert p.loc["amount", "q50"] == pytest.approx(1.5)
    assert p.loc["level", "distinct"] == 2
    assert p.loc["level", "top"] == "a (75%), b (25%)"


def test_profile_of_the_supplied_extract(loyalty_df):
    p = profile_frame(loyalty_df.drop(columns="customer_id"))
    assert p.loc["first_month", "missing"] == 125
    assert p.loc["joining_month", "missing"] == 125
    assert p.loc["spend", "min"] > 0
    assert p.loc["spend", "q05"] <= p.loc["spend", "q50"] <= p.loc["spend", "q95"]


def test_top_levels_limits_the_count():
    s = pd.Series(list("aaabbc"))
    assert top_levels(s, 1) == "a (50%)"
    assert top_levels(pd.Series([None, None])) == ""
