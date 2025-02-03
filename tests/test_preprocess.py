import numpy as np
import pandas as pd
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.features.preprocess import (
    feature_frame,
    make_preprocessor,
    original_feature,
    split_xy,
)


@pytest.mark.parametrize("kind", ["onehot", "ordinal", "spline"])
def test_preprocessors_produce_finite_matrices(train_df, kind):
    X, _ = split_xy(train_df)
    Z = make_preprocessor(kind).fit_transform(X)
    assert Z.shape[0] == len(X)
    assert np.isfinite(np.asarray(Z, float)).all()


def test_missing_numeric_gets_indicator(train_df):
    X, _ = split_xy(train_df)
    X = X.copy()
    X.loc[X.index[:5], "first_month"] = np.nan
    prep = make_preprocessor("onehot").fit(X)
    names = list(prep.get_feature_names_out())
    assert "missingindicator_first_month" in names


def test_unknown_category_at_scoring_time(train_df):
    X, _ = split_xy(train_df)
    prep = make_preprocessor("onehot").fit(X)
    new = X.head(2).copy()
    new["region"] = pd.Categorical(["Unknown", "Unknown"])
    assert prep.transform(new).shape[0] == 2


def test_original_feature_mapping():
    assert original_feature("loyalty_years_10+") == "loyalty_years"
    assert original_feature("missingindicator_first_month") == "first_month"
    assert original_feature("first_month") == "first_month"
    assert original_feature("items_in_first_month_sp_2") == "items_in_first_month"


def test_split_and_feature_frame(train_df):
    X, y = split_xy(train_df)
    assert list(X.columns) == FEATURES and len(y) == len(X)
    with pytest.raises(ValueError):
        split_xy(train_df.drop(columns=["spend"]))
    with pytest.raises(ValueError):
        feature_frame(train_df.drop(columns=["region"]))


def test_invalid_encoding():
    with pytest.raises(ValueError):
        make_preprocessor("hashing")
