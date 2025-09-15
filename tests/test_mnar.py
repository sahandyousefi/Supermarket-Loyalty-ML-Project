import numpy as np

from loyalty_spend.robustness.mnar import delta_sensitivity


def test_shifts_move_the_mean_of_the_column(loyalty_df):
    X = loyalty_df.drop(columns=["customer_id", "spend"])
    share = X["first_month"].isna().mean()
    table = delta_sensitivity(
        X, lambda d: float(d["first_month"].mean()), deltas=(-4.0, 0.0, 4.0), m=3
    )
    slope = np.polyfit(table["delta"], table["estimate"], 1)[0]
    # Shifting the imputed values by delta moves the mean by about share * delta.
    assert abs(slope - share) < 0.02
    assert (table["n_missing"] == X["first_month"].isna().sum()).all()
