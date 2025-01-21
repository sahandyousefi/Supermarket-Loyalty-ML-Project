import pytest

from loyalty_spend.config import Config, load_config


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
