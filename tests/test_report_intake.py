import json

from loyalty_spend.config import Config, PathsConfig
from loyalty_spend.reporting.report import build_report


def test_intake_section(raw_dir, tmp_path):
    cfg = Config(paths=PathsConfig(raw_dir=str(raw_dir), report_dir=str(tmp_path)))
    text = build_report(cfg, sections=("intake",)).read_text()
    assert "## Intake checks" in text
    summary = json.loads((tmp_path / "summary.json").read_text())["intake"]
    files = {row["file"]: row for row in summary["files"]}
    assert files["loyalty.csv"]["exact_duplicate_rows"] == 2
    assert files["train.csv"]["duplicate_id_rows"] == 0
    assert all(row["max_abs_z"] > 6 for row in summary["outliers"])
