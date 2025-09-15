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


def test_rubin_rules_by_hand():
    import pytest

    from loyalty_spend.robustness.imputation import pool_rubin

    est = np.array([[1.0], [1.2], [0.8]])
    var = np.array([[0.04], [0.04], [0.04]])
    out = pool_rubin(est, var).iloc[0]
    b = 0.04
    t = 0.04 + (1 + 1 / 3) * b
    assert out["estimate"] == pytest.approx(1.0)
    assert out["se"] == pytest.approx(np.sqrt(t))
    assert out["missing_information"] == pytest.approx((4 / 3) * b / t)


def test_pooled_intervals_cover_under_missingness():
    import pandas as pd

    from loyalty_spend.robustness.imputation import pooled_ols

    rng = np.random.default_rng(0)
    covered = 0
    for _ in range(20):
        n = 400
        x = rng.normal(10, 2, n)
        g = rng.choice(["a", "b"], n)
        y = 1.0 + 0.5 * x + (g == "b") * 2.0 + rng.normal(size=n)
        x_obs = np.where(rng.uniform(size=n) < 0.3, np.nan, x)
        frame = pd.DataFrame({"first_month": x_obs, "region": g})
        sets = []
        for _ in range(5):
            filled = x_obs.copy()
            miss = np.isnan(filled)
            # Proper draws from the regression of x on y and g.
            A = np.column_stack([np.ones((~miss).sum()), y[~miss], g[~miss] == "b"])
            beta, *_ = np.linalg.lstsq(A, x[~miss], rcond=None)
            sd = np.std(x[~miss] - A @ beta)
            Am = np.column_stack([np.ones(miss.sum()), y[miss], g[miss] == "b"])
            filled[miss] = Am @ beta + rng.normal(scale=sd, size=miss.sum())
            sets.append(frame.assign(first_month=filled))
        table = pooled_ols(sets, y, ["first_month", "region"]).set_index("term")
        covered += (
            table.loc["first_month", "ci_lower"]
            < 0.5
            < table.loc["first_month", "ci_upper"]
        )
    assert covered >= 16
