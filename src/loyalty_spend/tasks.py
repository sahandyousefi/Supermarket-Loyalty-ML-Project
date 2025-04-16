"""The four deliverables of the original brief, produced by the package.

Task 1  ``clean_data``: loyalty.csv cleaned to the data dictionary.
Task 2  ``spend_by_years``: mean and variance of spend by tenure band.
Task 3  ``base_result``: baseline predictions for test.csv.
Task 4  ``compare_result``: comparison-model predictions for test.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from loyalty_spend.config import Config, load_config
from loyalty_spend.data.io import load_clean, load_modeling_data
from loyalty_spend.data.schema import ID, TARGET
from loyalty_spend.features.preprocess import feature_frame, split_xy
from loyalty_spend.models.registry import build_model


def task1_clean_data(cfg: Config) -> pd.DataFrame:
    clean, _ = load_clean(cfg.paths.raw(cfg.paths.loyalty_file))
    return clean


def task2_spend_by_years(clean: pd.DataFrame) -> pd.DataFrame:
    out = (
        clean.groupby("loyalty_years", observed=True)[TARGET]
        .agg(avg_spend="mean", var_spend="var")
        .reset_index()
    )
    out[["avg_spend", "var_spend"]] = out[["avg_spend", "var_spend"]].round(2)
    out["loyalty_years"] = out["loyalty_years"].astype(str)
    return out


def _predict(cfg: Config, model_name: str) -> pd.DataFrame:
    train, test = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    X, y = split_xy(train)
    model = build_model(model_name).fit(X, y)
    pred = model.predict(feature_frame(test))
    return pd.DataFrame({ID: test[ID].to_numpy(), TARGET: pred.round(2)})


def task3_base_result(cfg: Config) -> pd.DataFrame:
    """Ordinary least squares, the baseline."""
    return _predict(cfg, "ols")


def task4_compare_result(cfg: Config, model_name: str = "lightgbm") -> pd.DataFrame:
    """A gradient-boosted comparison model."""
    return _predict(cfg, model_name)


def run_all(cfg: Config | None = None, output_dir: str | Path | None = None) -> dict:
    """Write the four deliverables as CSV files and return them."""
    cfg = cfg or load_config()
    out = Path(output_dir or cfg.paths.output_dir) / "tasks"
    out.mkdir(parents=True, exist_ok=True)
    clean = task1_clean_data(cfg)
    results = {
        "clean_data": clean,
        "spend_by_years": task2_spend_by_years(clean),
        "base_result": task3_base_result(cfg),
        "compare_result": task4_compare_result(cfg),
    }
    for name, frame in results.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return results
