import json

import pandas as pd
import pytest

from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.simulate import simulate_promotion
from loyalty_spend.causal.targeting import analyse_targeting
from loyalty_spend.cli import main


def test_analysis_on_a_heterogeneous_study(train_df):
    study = simulate_promotion(train_df, n=3000, effect=2.0, heterogeneity=3.0, seed=25)
    result = analyse_targeting(from_frame(study.frame), "t", cost=2.0, n_folds=3)
    assert result.blp["heterogeneity_p_value"] < 1e-4
    policies = result.policies.set_index("policy")
    assert (
        policies.loc["predicted profit above cost", "value"]
        > policies.loc["everybody", "value"]
    )
    assert result.qini > 0
    with pytest.raises(KeyError):
        analyse_targeting(from_frame(study.frame), "z")


def test_uplift_command(tmp_path, capsys):
    out = tmp_path / "uplift"
    assert main(["uplift", "--learner", "t", "--output-dir", str(out)]) == 0
    assert len(pd.read_csv(out / "policies.csv")) == 3
    summary = json.loads((out / "uplift_summary.json").read_text())
    assert {"blp", "gates", "qini"} <= set(summary)
    assert "heterogeneity" in capsys.readouterr().out
