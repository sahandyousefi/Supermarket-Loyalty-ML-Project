import json

from loyalty_spend.config import Config, PathsConfig
from loyalty_spend.reporting.report import build_report


def test_promotion_section(raw_dir, tmp_path):
    cfg = Config(paths=PathsConfig(raw_dir=str(raw_dir), report_dir=str(tmp_path)))
    path = build_report(cfg, sections=("promotion",))
    assert "## Effect of the sign-up promotion" in path.read_text()
    assert (tmp_path / "figures" / "promotion_effect.png").stat().st_size > 1000
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["promotion"]["estimates"]) == 8
