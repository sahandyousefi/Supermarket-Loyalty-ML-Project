"""End-to-end analysis that produces the results tables and figures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from loyalty_spend.analysis.drivers import (
    effect_sizes,
    lmg_importance,
    spend_by_group,
    tukey_table,
    welch_anova,
)
from loyalty_spend.business.rewards import RewardSchedule, forecast_liability
from loyalty_spend.config import Config
from loyalty_spend.data.io import load_clean, load_modeling_data
from loyalty_spend.data.lineage import compare_extracts
from loyalty_spend.data.schema import FEATURES, ID
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.cv import compare_results, cross_validate, make_splits
from loyalty_spend.evaluation.simulation_study import run_study
from loyalty_spend.features.preprocess import feature_frame, split_xy
from loyalty_spend.models.registry import available_models, build_model, model_family
from loyalty_spend.reporting import figures
from loyalty_spend.uncertainty.conformal import (
    ConformalizedQuantileRegressor,
    SplitConformalRegressor,
    coverage_by_group,
)

DEFAULT_MODELS = (
    "ols",
    "ridge",
    "elastic_net",
    "huber",
    "gamma_glm",
    "spline_ridge",
    "random_forest",
    "hist_gbm",
    "xgboost",
    "lightgbm",
    "mlp",
    "mlp_plr",
    "ft_transformer",
)
STUDY_MODELS = ("ols", "spline_ridge", "hist_gbm", "lightgbm", "mlp", "mlp_plr")


@dataclass(frozen=True)
class ReportSettings:
    models: Sequence[str] = DEFAULT_MODELS
    study_models: Sequence[str] = STUDY_MODELS
    study_n: int = 2000
    study_splits: int = 5
    conformal_scenario: str = "noisy"
    conformal_test_n: int = 5000


def _md_table(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt(v) -> str:
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(False)]
    return "\n".join([head, sep, *rows])


def driver_section(cfg: Config, fig_dir: Path) -> tuple[str, dict]:
    clean, log = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    by_years = spend_by_group(clean, "loyalty_years", seed=cfg.cv.seed)
    anova = welch_anova(clean, "loyalty_years")
    sizes = effect_sizes(clean, "loyalty_years")
    tukey = tukey_table(clean, "loyalty_years")
    shares = lmg_importance(clean, FEATURES)
    figures.plot_group_means(by_years, "loyalty_years", fig_dir / "spend_by_tenure.png")
    figures.plot_driver_shares(shares, fig_dir / "driver_shares.png")
    text = "\n\n".join(
        [
            "## Data quality (loyalty.csv)",
            _md_table(log.to_frame()),
            "## Spend by loyalty tenure",
            _md_table(by_years, 2),
            f"Welch ANOVA: F = {anova['f']:.1f} on ({anova['df1']:.0f}, "
            f"{anova['df2']:.1f}) df, p = {anova['p_value']:.2e}. "
            f"Eta squared {sizes['eta_squared']:.3f}, "
            f"omega squared {sizes['omega_squared']:.3f}.",
            "### Tukey HSD",
            _md_table(tukey, 3),
            "### Shapley decomposition of R squared",
            _md_table(shares),
        ]
    )
    return text, {"welch": anova, **sizes, "lmg": shares.to_dict("records")}


def lineage_section(cfg: Config) -> str:
    loyalty, _ = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    table = compare_extracts(loyalty, train)
    return "## Lineage check: loyalty.csv against train.csv\n\n" + _md_table(table, 3)


def benchmark_section(
    cfg: Config, settings: ReportSettings, fig_dir: Path
) -> tuple[str, pd.DataFrame, np.ndarray]:
    train, _ = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    X, y = split_xy(train)
    splits = make_splits(len(y), cfg.cv.n_splits, cfg.cv.n_repeats, cfg.cv.seed)
    names = [m for m in settings.models if m in available_models()]
    results = [cross_validate(m, build_model(m), X, y, splits) for m in names]
    table = compare_results(results, reference="ols")
    table.insert(1, "family", [model_family(m) for m in table["model"]])
    figures.plot_model_comparison(table, fig_dir / "model_comparison.png")
    cols = ["model", "family", "rmse_mean", "rmse_std", "mae_mean", "r2_mean"]
    cols += ["rmse_diff_vs_ols", "p_value"]
    text = "## Benchmark on train.csv\n\n" + _md_table(table[cols])
    oof = {r.name: r.oof for r in results}
    residuals = y.to_numpy() - oof["ols"]
    return text, table, residuals


def study_section(cfg: Config, settings: ReportSettings, fig_dir: Path) -> str:
    train, _ = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    table = run_study(
        train,
        settings.study_models,
        n=settings.study_n,
        n_splits=settings.study_splits,
        seed=cfg.cv.seed,
    )
    figures.plot_simulation_study(table, fig_dir / "simulation_study.png")
    wide = table.pivot(index="model", columns="scenario", values="rmse")
    wide = wide.reindex(index=list(settings.study_models), columns=list(SCENARIOS))
    bayes = table.groupby("scenario")["bayes_rmse"].first().reindex(wide.columns)
    wide.loc["bayes error"] = bayes
    return "## Simulation study (RMSE)\n\n" + _md_table(wide.reset_index(), 3)


def conformal_section(cfg: Config, settings: ReportSettings, fig_dir: Path) -> str:
    train, _ = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    scenario = SCENARIOS[settings.conformal_scenario]
    fit = simulate(train, scenario, n=settings.study_n, seed=cfg.cv.seed).frame
    test = simulate(
        train, scenario, n=settings.conformal_test_n, seed=cfg.cv.seed + 1
    ).frame
    X, y = split_xy(fit)
    X_test, y_test = split_xy(test)
    alpha = cfg.conformal.alpha
    methods = {
        "split conformal (OLS)": SplitConformalRegressor(
            build_model("ols"), alpha, cfg.conformal.calibration_size
        ).fit(X, y),
        "CQR (gradient boosting)": ConformalizedQuantileRegressor(
            alpha, cfg.conformal.calibration_size
        ).fit(X, y),
    }
    point = build_model("ols").fit(X, y).predict(X_test)
    deciles = pd.qcut(point, 10, labels=False) + 1
    rows, frames = [], []
    for name, method in methods.items():
        iv = method.predict_interval(X_test)
        rows.append(
            {
                "method": name,
                "coverage": iv.coverage(y_test),
                "mean_width": float(iv.width.mean()),
            }
        )
        frames.append(coverage_by_group(iv, y_test, deciles).assign(method=name))
    by_decile = pd.concat(frames, ignore_index=True)
    figures.plot_conditional_coverage(
        by_decile, 1 - alpha, fig_dir / "conditional_coverage.png"
    )
    worst = by_decile.groupby("method")["coverage"].min().rename("worst_decile")
    summary = pd.DataFrame(rows).merge(worst, on="method")
    return (
        f"## Prediction intervals ({settings.conformal_scenario} scenario, "
        f"target coverage {1 - alpha:.0%})\n\n" + _md_table(summary, 3)
    )


def liability_section(cfg: Config, residuals: np.ndarray, fig_dir: Path) -> str:
    train, test = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    X, y = split_xy(train)
    model = build_model("ols").fit(X, y)
    schedule = RewardSchedule(cfg.rewards.tiers)
    fc = forecast_liability(
        test[ID],
        model.predict(feature_frame(test)),
        residuals,
        schedule,
        cfg.rewards.n_simulations,
        cfg.rewards.seed,
    )
    s = fc.summary()
    figures.plot_liability(
        fc.total_draws, s["plug_in_total"], fig_dir / "liability.png"
    )
    near = int((fc.per_customer["tier_change_probability"] >= 0.2).sum())
    table = pd.DataFrame([{k: round(v, 2) for k, v in s.items()}])
    return (
        f"## Reward liability for the {len(test)} customers in test.csv\n\n"
        + _md_table(table, 2)
        + f"\n\nCustomers with at least a 20% chance of changing tier: {near}."
    )
