import json

from loyalty_spend.config import Config, PathsConfig
from loyalty_spend.reporting.report import build_report


def test_driver_section_reports_pairwise_and_partial_tests(raw_dir, tmp_path):
    cfg = Config(paths=PathsConfig(raw_dir=str(raw_dir), report_dir=str(tmp_path)))
    text = build_report(cfg, sections=("drivers",)).read_text()
    assert "### Games-Howell" in text and "Freedman-Lane" in text
    summary = json.loads((tmp_path / "summary.json").read_text())["drivers"]
    low, high = summary["omega_squared_interval"]
    assert low <= summary["omega_squared"] <= high
    assert len(summary["games_howell"]) == 10
