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


def test_robust_outliers_find_planted_errors(train_df):
    from loyalty_spend.data.intake import robust_outliers

    df = train_df.copy()
    numeric = ["spend", "first_month"]
    categorical = ["loyalty_years", "joining_month", "region", "promotion"]
    # A spend typed ten times too large and a first month in the wrong unit.
    df.loc[5, "spend"] *= 10
    df.loc[9, "first_month"] *= 100
    flagged = robust_outliers(df, numeric, categorical)
    top = set(flagged.head(2)["customer_id"])
    assert top == {df.loc[5, "customer_id"], df.loc[9, "customer_id"]}
    planted = flagged.head(2)["max_abs_z"]
    assert (planted > 50).all()
    # Apart from the two planted errors only a handful of members stand out.
    assert flagged["flagged"].sum() <= 4
