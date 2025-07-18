import json

import pandas as pd

from loyalty_spend.cli import main


def test_design_command(cfg_small, tmp_path, capsys):
    out = tmp_path / "design"
    args = ["design", "--config", str(cfg_small), "--offsets", "-5", "0", "5"]
    assert main([*args, "--output-dir", str(out)]) == 0
    assert len(pd.read_csv(out / "schedules.csv")) == 2
    risk = json.loads((out / "risk.json").read_text())
    assert risk["expected_shortfall"] >= risk["var"]
    assert "value at risk" in capsys.readouterr().out
