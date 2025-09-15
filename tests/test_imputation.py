import numpy as np

from loyalty_spend.robustness.imputation import multiple_imputations


def test_imputations_fill_only_the_gaps_and_vary(loyalty_df):
    X = loyalty_df.drop(columns=["customer_id", "spend"])
    gaps = X["first_month"].isna().to_numpy()
    assert gaps.sum() > 50
    sets = multiple_imputations(X, m=3)
    for d in sets:
        assert not d["first_month"].isna().any()
        np.testing.assert_allclose(
            d.loc[~gaps, "first_month"], X.loc[~gaps, "first_month"].to_numpy()
        )
        assert (d["items_in_first_month"] == d["items_in_first_month"].round()).all()
    assert not np.allclose(
        sets[0].loc[gaps, "first_month"], sets[1].loc[gaps, "first_month"]
    )


def test_imputed_values_follow_the_conditional_relation(loyalty_df):
    X = loyalty_df.drop(columns=["customer_id", "spend"])
    d = multiple_imputations(X, m=1)[0]
    gaps = X["first_month"].isna().to_numpy()
    observed = X.loc[~gaps, "first_month"]
    imputed = d.loc[gaps, "first_month"]
    assert abs(imputed.mean() - observed.mean()) < observed.std()
