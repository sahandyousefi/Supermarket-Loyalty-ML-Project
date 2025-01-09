import numpy as np
import pandas as pd

from loyalty_spend.data.schema import COLUMNS, allowed_values, validate


def _frame(**overrides):
    base = {
        "customer_id": [1, 2],
        "spend": [120.5, 130.0],
        "first_month": [15.0, 20.0],
        "items_in_first_month": [5, 9],
        "region": ["Europe", "Americas"],
        "loyalty_years": ["0-1", "10+"],
        "joining_month": ["Jan", "Unknown"],
        "promotion": ["Yes", "No"],
    }
    base.update(overrides)
    return pd.DataFrame(base)[COLUMNS]


def test_valid_frame_passes():
    assert validate(_frame()).ok


def test_unknown_level_is_an_error():
    report = validate(_frame(region=["Europe", "Mars"]))
    assert not report.ok and "Mars" in report.errors[0]


def test_duplicate_ids_and_negative_amounts_are_errors():
    report = validate(_frame(customer_id=[1, 1], spend=[-1.0, 10.0]))
    assert len(report.errors) == 2


def test_missing_values_are_warnings():
    report = validate(_frame(first_month=[np.nan, 20.0]))
    assert report.ok and report.warnings


def test_scoring_frame_without_target():
    assert validate(_frame().drop(columns=["spend"]), require_target=False).ok


def test_unknown_level_only_where_dictionary_allows_it():
    assert "Unknown" in allowed_values("region")
    assert "Unknown" not in allowed_values("loyalty_years")
