import numpy as np
import pytest

from loyalty_spend.segments.profiles import profile_segments


def test_profiles_of_known_groups(train_df):
    X = train_df.drop(columns=["customer_id", "spend"]).reset_index(drop=True)
    labels = (X["region"].astype(str) == "Europe").to_numpy(int)
    table = profile_segments(X, labels, spend=train_df["spend"]).set_index("segment")
    assert table["size"].sum() == len(X)
    assert table["share"].sum() == pytest.approx(1.0)
    assert table.loc[1, "region_mode"] == "Europe"
    assert table.loc[1, "region_share"] == 1.0
    assert table.loc[1, "region_lift"] > 1.0
    europe = train_df["region"].astype(str).to_numpy() == "Europe"
    assert table.loc[1, "spend_mean"] == pytest.approx(train_df["spend"][europe].mean())
    assert np.isfinite(table["first_month_median"]).all()
