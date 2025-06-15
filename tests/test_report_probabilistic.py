import json

import pytest

from loyalty_spend.config import Config, PathsConfig
from loyalty_spend.reporting.report import ReportSettings, build_report


@pytest.mark.slow
def test_probabilistic_section(raw_dir, tmp_path):
    cfg = Config(paths=PathsConfig(raw_dir=str(raw_dir), report_dir=str(tmp_path)))
    settings = ReportSettings(study_n=600, study_splits=3, conformal_test_n=600)
    path = build_report(cfg, settings, sections=("probabilistic",))
    text = path.read_text()
    assert "## Probabilistic forecasts" in text
    assert (tmp_path / "figures" / "quantile_reliability.png").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["probabilistic"]["crps"]) == 5
    assert len(summary["probabilistic"]["intervals"]) == 4
