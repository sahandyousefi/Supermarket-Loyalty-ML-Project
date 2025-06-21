import json

import pandas as pd

from loyalty_spend.cli import main


def test_effect_command_on_the_training_file(tmp_path, capsys):
    out = tmp_path / "effect"
    assert main(["effect", "--n-boot", "50", "--output-dir", str(out)]) == 0
    table = pd.read_csv(out / "effect_estimates.csv")
    assert len(table) == 8
    diagnostics = json.loads((out / "effect_diagnostics.json").read_text())
    assert {"overlap", "sensitivity"} <= set(diagnostics)
    text = capsys.readouterr().out
    assert "robustness value" in text and "treated" in text
