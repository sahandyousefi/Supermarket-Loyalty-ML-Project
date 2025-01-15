import pandas as pd

from loyalty_spend.data.cleaning import (
    coerce_numeric,
    normalize_category,
)


def test_coerce_numeric_separates_placeholders_from_garbage():
    values, n_placeholder, n_invalid = coerce_numeric(
        pd.Series(["1.5", ".", "abc", None])
    )
    assert values.isna().sum() == 3
    assert (n_placeholder, n_invalid) == (1, 1)


def test_normalize_category_maps_case_variants():
    values, changed = normalize_category(
        pd.Series(["YES", " no", "Yes", "maybe"]), ["Yes", "No"]
    )
    assert values.tolist()[:3] == ["Yes", "No", "Yes"]
    assert pd.isna(values.iloc[3])
    assert changed == 2
