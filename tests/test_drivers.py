import numpy as np
import pandas as pd
import pytest
from scipy import stats

from loyalty_spend.analysis.drivers import (
    spend_by_group,
    welch_anova,
)


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
