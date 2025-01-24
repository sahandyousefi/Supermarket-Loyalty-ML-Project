import numpy as np
import pandas as pd
import pytest
from scipy import stats

from loyalty_spend.analysis.drivers import (
    effect_sizes,
    lmg_importance,
    spend_by_group,
    tukey_table,
    welch_anova,
)
from loyalty_spend.data.schema import FEATURES


@pytest.fixture
def groups():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "g": np.repeat(["a", "b", "c"], [40, 60, 80]),
            "spend": np.concatenate(
                [rng.normal(10, 1, 40), rng.normal(11, 2, 60), rng.normal(10, 3, 80)]
            ),
        }
    )


def test_group_table_and_interval(groups):
    t = spend_by_group(groups, "g", n_boot=500)
    assert list(t["n"]) == [40, 60, 80]
    assert ((t["ci_low"] <= t["avg_spend"]) & (t["avg_spend"] <= t["ci_high"])).all()


def test_welch_reduces_to_classic_anova_for_two_equal_variance_groups():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": np.repeat(["a", "b"], 50), "spend": rng.normal(0, 1, 100)})
    df.loc[df.g == "b", "spend"] += 0.5
    welch = welch_anova(df, "g")
    t = stats.ttest_ind(
        df.loc[df.g == "a", "spend"], df.loc[df.g == "b", "spend"], equal_var=False
    )
    assert welch["p_value"] == pytest.approx(t.pvalue, rel=1e-6)


def test_effect_sizes_are_bounded(groups):
    sizes = effect_sizes(groups, "g")
    assert 0 <= sizes["omega_squared"] <= sizes["eta_squared"] <= 1


def test_tukey_pairs(groups):
    t = tukey_table(groups, "g")
    assert len(t) == 3
    assert ((t["ci_low"] <= t["diff"]) & (t["diff"] <= t["ci_high"])).all()


def test_lmg_shares_add_up_to_r2(loyalty_df):
    from sklearn.linear_model import LinearRegression

    data = loyalty_df.dropna()
    shares = lmg_importance(data, FEATURES)
    X = pd.get_dummies(data[FEATURES], drop_first=True, dtype=float)
    r2 = LinearRegression().fit(X, data["spend"]).score(X, data["spend"])
    assert shares["r2_share"].sum() == pytest.approx(r2, rel=1e-8)
    assert shares["feature"].iloc[0] == "loyalty_years"
