import pandas as pd

from loyalty_spend.data.intake import (
    duplicate_ids,
    exact_duplicates,
    intake_summary,
    near_duplicates,
)

NUMERIC = ["spend", "first_month"]


def _frame():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 3, 4, 5],
            "spend": [100.0, 120.0, 140.0, 141.0, 100.0, 100.01],
            "first_month": [20.0, 25.0, 30.0, 31.0, 20.0, 20.0],
            "region": ["A", "B", "A", "A", "A", "A"],
        }
    )


def test_duplicate_ids_and_records():
    df = _frame()
    assert list(duplicate_ids(df)["customer_id"]) == [3, 3]
    assert set(exact_duplicates(df)["customer_id"]) == {1, 4}


def test_near_duplicates_need_equal_categories_and_close_amounts():
    pairs = near_duplicates(_frame(), NUMERIC)
    found = {tuple(sorted(p)) for p in pairs[["id_a", "id_b"]].to_numpy()}
    # 3 and 3 differ by a full unit; 1 and 4 are exact duplicates.
    assert found == {(1, 5), (4, 5)}
    assert intake_summary(_frame(), NUMERIC) == {
        "rows": 6,
        "duplicate_id_rows": 2,
        "exact_duplicate_rows": 2,
        "near_duplicate_pairs": 2,
    }


def test_supplied_files(loyalty_df, train_df):
    numeric = ["spend", "first_month", "items_in_first_month"]
    assert intake_summary(train_df, numeric)["exact_duplicate_rows"] == 0
    s = intake_summary(loyalty_df, numeric)
    # Two members of loyalty.csv share every value but the id.
    assert s["duplicate_id_rows"] == 0 and s["exact_duplicate_rows"] == 2
