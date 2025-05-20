import pandas as pd

from loyalty_spend.cli import main


def test_supplied_files_pass_their_contracts(raw_dir, capsys):
    for name, contract in [("train.csv", "training"), ("test.csv", "scoring")]:
        assert main(["validate", "--input", str(raw_dir / name)]) == 0
        assert f"contract '{contract}' passed" in capsys.readouterr().out


def test_broken_extract_fails_with_a_report(raw_dir, tmp_path):
    raw = pd.read_csv(raw_dir / "train.csv", dtype=str)
    raw.loc[:4, "spend"] = "-5"
    raw.loc[5:9, "customer_id"] = raw.loc[0, "customer_id"]
    path = tmp_path / "broken.csv"
    raw.to_csv(path, index=False)
    out = tmp_path / "violations.csv"
    assert main(["validate", "--input", str(path), "--output", str(out)]) == 1
    table = pd.read_csv(out)
    rules = set(zip(table["column"], table["rule"]))
    assert ("customer_id", "unique") in rules
