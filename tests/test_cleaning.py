import pandas as pd

from loyalty_spend.data.cleaning import (
    clean_loyalty,
    coerce_numeric,
    normalize_category,
)
from loyalty_spend.data.io import read_raw
from loyalty_spend.data.schema import COLUMNS, LOYALTY_ORDER


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


def test_clean_loyalty_matches_dictionary(raw_dir):
    clean, report = clean_loyalty(read_raw(raw_dir / "loyalty.csv"))
    assert list(clean.columns) == COLUMNS
    assert not clean.isna().any().any()
    assert set(clean["promotion"]) == {"Yes", "No"}
    assert list(clean["loyalty_years"].cat.categories) == LOYALTY_ORDER
    assert clean["loyalty_years"].cat.ordered
    log = report.to_frame().set_index(["column", "rule"])["n"]
    assert log[("first_month", "placeholder to missing")] == 125
    assert log[("promotion", "case or whitespace normalized")] == 125


def test_spec_fill_values(raw_dir):
    raw = read_raw(raw_dir / "loyalty.csv")
    clean, _ = clean_loyalty(raw)
    was_missing = raw["joining_month"].isna()
    assert (clean.loc[was_missing, "joining_month"] == "Unknown").all()
    assert (clean.loc[raw["first_month"] == ".", "first_month"] == 0).all()
