import pytest

from loyalty_spend.config import Config, MonitoringConfig, load_config
from loyalty_spend.monitoring.drift import DriftThresholds


def test_default_file_matches_dataclass_defaults():
    assert load_config() == Config()


def test_partial_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("cv:\n  n_splits: 3\n")
    cfg = load_config(path)
    assert cfg.cv.n_splits == 3
    assert cfg.cv.n_repeats == Config().cv.n_repeats


def test_reward_tiers_become_tuples(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("rewards:\n  tiers:\n    - [0, 0.0]\n    - [50, 0.1]\n")
    assert load_config(path).rewards.tiers == ((0.0, 0.0), (50.0, 0.1))


@pytest.mark.parametrize(
    "text",
    [
        "unknown: 1\n",
        "cv:\n  folds: 3\n",
        "cv:\n  n_splits: 1\n",
        "conformal:\n  alpha: 1.5\n",
    ],
)
def test_invalid_config_raises(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "absent.yaml") == Config()


def test_monitoring_section_defaults_and_validation(tmp_path):
    cfg = load_config()
    assert cfg.monitoring == MonitoringConfig()
    t = DriftThresholds.from_config(cfg.monitoring)
    assert (t.psi_moderate, t.psi_large, t.fdr) == (0.1, 0.25, 0.05)

    path = tmp_path / "bad.yaml"
    path.write_text("monitoring:\n  psi_moderate: 0.3\n  psi_large: 0.2\n")
    with pytest.raises(ValueError, match="psi_moderate"):
        load_config(path)
