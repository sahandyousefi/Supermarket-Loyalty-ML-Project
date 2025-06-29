import pandas as pd

from loyalty_spend.cli import main


def test_segments_command(raw_dir, tmp_path, capsys):
    out = tmp_path / "seg"
    args = ["segments", "--input", str(raw_dir / "test.csv"), "--max-k", "4"]
    assert main([*args, "--n-boot", "5", "--output-dir", str(out)]) == 0
    assignments = pd.read_csv(out / "assignments.csv")
    assert len(assignments) == 250
    profiles = pd.read_csv(out / "profiles.csv")
    assert profiles["size"].sum() == 250
    assert set(pd.read_csv(out / "silhouette.csv")["k"]) == {2, 3, 4}
    assert "segments" in capsys.readouterr().out
