import pandas as pd
import pytest

from loyalty_spend.cli import build_parser, main
from loyalty_spend.config import load_config
from loyalty_spend.tasks import (
    task1_clean_data,
    task2_spend_by_years,
    task3_base_result,
)


@pytest.fixture
def cfg(raw_dir, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        f"paths:\n  raw_dir: {raw_dir}\n  output_dir: {tmp_path / 'art'}\n"
        f"  report_dir: {tmp_path / 'rep'}\n"
        "cv:\n  n_splits: 3\n  n_repeats: 1\n"
        "rewards:\n  n_simulations: 200\n"
    )
    return path


def test_task_outputs(cfg):
    c = load_config(cfg)
    clean = task1_clean_data(c)
    years = task2_spend_by_years(clean)
    assert list(years.columns) == ["loyalty_years", "avg_spend", "var_spend"]
    assert list(years["loyalty_years"]) == ["0-1", "1-3", "3-5", "5-10", "10+"]
    base = task3_base_result(c)
    assert list(base.columns) == ["customer_id", "spend"] and len(base) == 250


def test_parser_lists_commands():
    text = build_parser().format_help()
    for name in ["clean", "drivers", "compare", "train", "predict", "rewards", "tasks"]:
        assert name in text


def test_train_predict_rewards_cycle(cfg, raw_dir, tmp_path, capsys):
    model = tmp_path / "m.joblib"
    assert (
        main(
            ["train", "--config", str(cfg), "--model", "ridge", "--output", str(model)]
        )
        == 0
    )
    out = tmp_path / "pred.csv"
    assert (
        main(["predict", str(raw_dir / "test.csv"), str(out), "--model", str(model)])
        == 0
    )
    assert len(pd.read_csv(out)) == 250
    assert (
        main(
            [
                "rewards",
                "--config",
                str(cfg),
                str(raw_dir / "test.csv"),
                "--model",
                str(model),
            ]
        )
        == 0
    )
    assert "expected_total" in capsys.readouterr().out


def test_clean_and_compare_commands(cfg, tmp_path):
    out = tmp_path / "clean.csv"
    assert main(["clean", "--config", str(cfg), "--output", str(out)]) == 0
    assert len(pd.read_csv(out)) == 1246
    table = tmp_path / "cmp.csv"
    assert (
        main(
            [
                "compare",
                "--config",
                str(cfg),
                "--models",
                "ols",
                "ridge",
                "--output",
                str(table),
            ]
        )
        == 0
    )
    assert set(pd.read_csv(table)["model"]) == {"ols", "ridge"}


def test_version_flag(capsys):
    import pytest

    from loyalty_spend import __version__
    from loyalty_spend.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_verbosity_sets_the_log_level():
    import logging

    from loyalty_spend.cli import configure_logging

    for count, level in [(0, logging.WARNING), (1, logging.INFO), (3, logging.DEBUG)]:
        configure_logging(count)
        assert logging.getLogger().level == level
    configure_logging(0)


def test_rewards_with_heteroscedastic_uncertainty(cfg, raw_dir, tmp_path, capsys):
    out = tmp_path / "rewards.csv"
    args = ["rewards", "--config", str(cfg), str(raw_dir / "test.csv")]
    assert main([*args, "--uncertainty", "heteroscedastic", "--output", str(out)]) == 0
    table = pd.read_csv(out)
    assert len(table) == 250
    assert table["tier_change_probability"].between(0, 1).all()
    assert "expected_total" in capsys.readouterr().out


def test_predict_with_intervals_and_validation(cfg, raw_dir, tmp_path):
    model = tmp_path / "m.joblib"
    assert (
        main(["train", "--config", str(cfg), "--model", "ols", "--output", str(model)])
        == 0
    )
    out = tmp_path / "pred.csv"
    args = ["predict", str(raw_dir / "test.csv"), str(out), "--model", str(model)]
    assert main([*args, "--alpha", "0.1", "--validate"]) == 0
    scores = pd.read_csv(out)
    assert list(scores.columns) == ["customer_id", "spend", "lower", "upper"]
    assert (scores["lower"] <= scores["upper"]).all()


def test_monitor_command(cfg, raw_dir, tmp_path, capsys):
    import json

    model = tmp_path / "m.joblib"
    assert (
        main(["train", "--config", str(cfg), "--model", "ols", "--output", str(model)])
        == 0
    )
    out = tmp_path / "monitor.json"
    args = [
        "monitor",
        "--config",
        str(cfg),
        str(raw_dir / "test.csv"),
        "--model",
        str(model),
    ]
    assert main([*args, "--output", str(out)]) == 0
    assert json.loads(out.read_text())["status"] in {"ok", "warn"}
    assert "status:" in capsys.readouterr().out


def test_model_card_and_serve_commands(cfg, tmp_path, monkeypatch):
    import sys
    import types

    model = tmp_path / "m.joblib"
    assert (
        main(["train", "--config", str(cfg), "--model", "ols", "--output", str(model)])
        == 0
    )
    card = tmp_path / "card.md"
    assert main(["model-card", "--model", str(model), "--output", str(card)]) == 0
    assert card.read_text().startswith("# Model card: ols")

    calls = {}
    fake = types.SimpleNamespace(
        run=lambda app, host, port: calls.update(host=host, port=port)
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    assert (
        main(["serve", "--config", str(cfg), "--model", str(model), "--port", "9001"])
        == 0
    )
    assert calls == {"host": "127.0.0.1", "port": 9001}


def test_bayes_command(cfg, tmp_path, capsys):
    import json

    out = tmp_path / "bayes"
    args = [
        "bayes",
        "--config",
        str(cfg),
        "--chains",
        "2",
        "--iter",
        "300",
        "--warmup",
        "100",
    ]
    assert main([*args, "--output-dir", str(out)]) == 0
    assert "WAIC" in capsys.readouterr().out
    assert json.loads((out / "bayes.json").read_text())["cost"]["expected_total"] > 0


def test_next_tier_command(cfg, raw_dir, tmp_path, capsys):
    model = tmp_path / "m.joblib"
    assert (
        main(["train", "--config", str(cfg), "--model", "ols", "--output", str(model)])
        == 0
    )
    out = tmp_path / "next.csv"
    args = [
        "next-tier",
        "--config",
        str(cfg),
        str(raw_dir / "test.csv"),
        "--model",
        str(model),
    ]
    assert main([*args, "--output", str(out)]) == 0
    table = pd.read_csv(out)
    assert {"gap", "first_month_needed", "feasible"} <= set(table.columns)
    assert (table["gap"] >= 0).all()
    assert "below the top tier" in capsys.readouterr().out
