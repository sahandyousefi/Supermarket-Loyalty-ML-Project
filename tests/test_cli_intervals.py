import pandas as pd
import pytest

from loyalty_spend.cli import main


def test_intervals_command_writes_a_table(tmp_path, capsys):
    out = tmp_path / "intervals.csv"
    args = ["intervals", "--n-fit", "800", "--n-test", "800", "--output", str(out)]
    assert main(args) == 0
    table = pd.read_csv(out)
    assert len(table) == 4
    assert "target coverage 90%" in capsys.readouterr().out


def test_unknown_scenario():
    with pytest.raises(SystemExit):
        main(["intervals", "--scenario", "nope"])
