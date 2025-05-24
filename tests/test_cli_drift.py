import pandas as pd

from loyalty_spend.cli import main


def test_drift_between_supplied_files(raw_dir, tmp_path, capsys):
    out = tmp_path / "drift.csv"
    assert main(["drift", "--output", str(out), "--fail-on", "large"]) == 0
    report = pd.read_csv(out)
    assert set(report["feature"]) == {
        "first_month",
        "items_in_first_month",
        "region",
        "loyalty_years",
        "joining_month",
        "promotion",
    }
    assert "domain classifier AUC" in capsys.readouterr().out


def test_drift_fails_on_a_shifted_extract(raw_dir, tmp_path):
    raw = pd.read_csv(raw_dir / "test.csv", dtype=str)
    raw["first_month"] = (raw["first_month"].astype(float) * 1.6).round(2).astype(str)
    path = tmp_path / "shifted.csv"
    raw.to_csv(path, index=False)
    assert main(["drift", "--current", str(path), "--fail-on", "large"]) == 1
    assert main(["drift", "--current", str(path)]) == 0
