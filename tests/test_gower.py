import numpy as np
import pandas as pd
import pytest

from loyalty_spend.segments.gower import gower_matrix


def test_hand_computed_distances():
    X = pd.DataFrame(
        {
            "first_month": [10.0, 20.0, 30.0],
            "loyalty_years": ["0-1", "10+", "3-5"],
            "region": ["Europe", "Europe", "Americas"],
        }
    )
    D = gower_matrix(X, numeric=["first_month"])
    # Pair (0, 1): numeric 10/20, ordinal 4/4, nominal 0 -> mean (0.5 + 1 + 0) / 3.
    assert D[0, 1] == pytest.approx(0.5)
    # Pair (0, 2): numeric 1, ordinal 2/4, nominal 1 -> 2.5 / 3.
    assert D[0, 2] == pytest.approx(2.5 / 3)
    np.testing.assert_allclose(D, D.T)
    assert np.all(np.diag(D) == 0)


def test_missing_values_drop_the_feature_for_that_pair():
    X = pd.DataFrame(
        {"first_month": [10.0, np.nan], "region": ["Europe", "Asia/Pacific"]}
    )
    D = gower_matrix(X, numeric=["first_month"], ordinal={})
    assert D[0, 1] == pytest.approx(1.0)


def test_weights_and_range(loyalty_df):
    X = loyalty_df.drop(columns=["customer_id", "spend"]).head(200)
    D = gower_matrix(X)
    assert D.shape == (200, 200)
    assert np.nanmin(D) >= 0 and np.nanmax(D) <= 1
    only_region = gower_matrix(X, weights={c: 0.0 for c in X.columns if c != "region"})
    same = X["region"].to_numpy()[:, None] == X["region"].to_numpy()[None, :]
    np.testing.assert_allclose(only_region, 1.0 - same)
