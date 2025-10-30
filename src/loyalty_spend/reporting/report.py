"""End-to-end analysis that produces the results tables and figures."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from loyalty_spend.analysis.drivers import (
    effect_sizes,
    freedman_lane,
    games_howell,
    lmg_importance,
    omega_squared_interval,
    spend_by_group,
    tukey_table,
    welch_anova,
)
from loyalty_spend.bayes.checks import posterior_predictive_check
from loyalty_spend.bayes.compare import waic
from loyalty_spend.bayes.hierarchical import fit_hierarchical
from loyalty_spend.business.programme import design_programme
from loyalty_spend.business.rewards import RewardSchedule, forecast_liability
from loyalty_spend.causal.analysis import analyse_promotion
from loyalty_spend.causal.data import from_frame
from loyalty_spend.causal.targeting import analyse_targeting
from loyalty_spend.config import Config, load_config
from loyalty_spend.data.io import load_clean, load_modeling_data
from loyalty_spend.data.lineage import compare_extracts
from loyalty_spend.data.schema import FEATURES, ID
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.cv import compare_results, cross_validate, make_splits
from loyalty_spend.evaluation.probabilistic import (
    compare_distributions,
    cross_validate_distribution,
)
from loyalty_spend.evaluation.scoring import quantile_reliability
from loyalty_spend.evaluation.simulation_study import run_study
from loyalty_spend.features.preprocess import feature_frame, split_xy
from loyalty_spend.models.forecasters import FORECASTERS, TAU_GRID, build_forecaster
from loyalty_spend.models.registry import available_models, build_model, model_family
from loyalty_spend.reporting import figures
from loyalty_spend.reporting.markdown import markdown_table
from loyalty_spend.uncertainty.compare import compare_intervals
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


def driver_section(cfg: Config, fig_dir: Path) -> tuple[str, dict]:
    clean, log = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    by_years = spend_by_group(clean, "loyalty_years", seed=cfg.cv.seed)
    anova = welch_anova(clean, "loyalty_years")
    sizes = effect_sizes(clean, "loyalty_years")
    tukey = tukey_table(clean, "loyalty_years")
    gh = games_howell(clean, "loyalty_years")
    omega = omega_squared_interval(clean, "loyalty_years")
    others = ["region", "joining_month", "promotion"]
    partial = freedman_lane(clean, ["loyalty_years"], others, seed=cfg.cv.seed)
    shares = lmg_importance(clean, FEATURES)
    figures.plot_group_means(by_years, "loyalty_years", fig_dir / "spend_by_tenure.png")
    figures.plot_driver_shares(shares, fig_dir / "driver_shares.png")
    text = "\n\n".join(
        [
            "## Data quality (loyalty.csv)",
            markdown_table(log.to_frame()),
            "## Spend by loyalty tenure",
            markdown_table(by_years, 2),
            f"Welch ANOVA: F = {anova['f']:.1f} on ({anova['df1']:.0f}, "
            f"{anova['df2']:.1f}) df, p = {anova['p_value']:.2e}. "
            f"Eta squared {sizes['eta_squared']:.3f}, "
            f"omega squared {sizes['omega_squared']:.3f} "
            f"(95% interval {omega['low']:.3f} to {omega['high']:.3f}).",
            "### Tukey HSD",
            markdown_table(tukey, 3),
            "### Games-Howell",
            "Pairwise comparisons with the standard error of each pair, for "
            "unequal variances across tenure bands.",
            markdown_table(gh, 3),
            "### Tenure given the other attributes",
            f"Partial F = {partial['f']:.1f} on ({partial['df1']:.0f}, "
            f"{partial['df2']:.0f}) df for tenure added to region, joining month "
            f"and promotion; Freedman-Lane permutation p = "
            f"{partial['p_permutation']:.4f} from 2,000 permutations.",
            "### Shapley decomposition of R squared",
            markdown_table(shares),
        ]
    )
    return text, {
        "welch": anova,
        **sizes,
        "omega_squared_interval": [omega["low"], omega["high"]],
        "freedman_lane": partial,
        "games_howell": gh.to_dict("records"),
        "lmg": shares.to_dict("records"),
    }


def lineage_section(cfg: Config) -> str:
    loyalty, _ = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    table = compare_extracts(loyalty, train)
    return "## Lineage check: loyalty.csv against train.csv\n\n" + markdown_table(
        table, 3
    )


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
    text = "## Benchmark on train.csv\n\n" + markdown_table(table[cols])
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
    return "## Simulation study (RMSE)\n\n" + markdown_table(wide.reset_index(), 3)


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
    methods: dict[str, SplitConformalRegressor | ConformalizedQuantileRegressor] = {
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
        f"target coverage {1 - alpha:.0%})\n\n" + markdown_table(summary, 3)
    )


PROBABILISTIC_FORECASTERS = (
    "gaussian",
    "gaussian_heteroscedastic",
    "conformal_predictive",
    "linear_quantile",
    "gbm_quantile",
)


def probabilistic_section(
    cfg: Config, settings: ReportSettings, fig_dir: Path
) -> tuple[str, dict]:
    """CRPS comparison, quantile reliability and interval methods.

    Uses the same heteroscedastic scenario as the conformal section, because
    on the nearly noise-free supplied data every predictive distribution is
    a narrow spike and the comparison carries no information.
    """
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
    splits = make_splits(len(y), settings.study_splits, 1, cfg.cv.seed)
    results = [
        cross_validate_distribution(name, FORECASTERS[name], X, y, splits)
        for name in PROBABILISTIC_FORECASTERS
    ]
    crps = compare_distributions(results, "gaussian")
    shares = {}
    for name in PROBABILISTIC_FORECASTERS:
        q = build_forecaster(name).fit(X, y).quantiles(X_test)
        shares[name] = quantile_reliability(y_test, q, TAU_GRID)
    figures.plot_quantile_reliability(
        np.array(TAU_GRID), shares, fig_dir / "quantile_reliability.png"
    )
    intervals = compare_intervals(
        X,
        y,
        X_test,
        y_test,
        alpha=cfg.conformal.alpha,
        calibration_size=cfg.conformal.calibration_size,
        seed=cfg.cv.seed,
    )
    crps_cols = [
        "model",
        "crps_mean",
        "crps_diff",
        "p_value",
        "coverage_mean",
        "interval_score_mean",
        "pit_chi2_p_value",
    ]
    iv_cols = [
        "method",
        "coverage",
        "mean_width",
        "interval_score",
        "worst_group_coverage",
        "best_group_coverage",
    ]
    text = "\n\n".join(
        [
            f"## Probabilistic forecasts ({settings.conformal_scenario} scenario)",
            "Cross-validated CRPS on a grid of 19 quantile levels; the 90% "
            "interval uses the 5% and 95% levels.",
            markdown_table(crps[crps_cols], 3),
            "### Interval methods on new customers",
            markdown_table(intervals[iv_cols], 3),
        ]
    )
    summary = {
        "crps": crps[crps_cols].to_dict("records"),
        "intervals": intervals[iv_cols].to_dict("records"),
    }
    return text, summary


def promotion_section(cfg: Config, fig_dir: Path) -> tuple[str, dict]:
    """Observational estimates of the promotion effect on train.csv."""
    train, _ = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    result = analyse_promotion(from_frame(train), seed=cfg.cv.seed)
    figures.plot_effect_estimates(result.estimates, fig_dir / "promotion_effect.png")
    cols = ["method", "estimand", "estimate", "se", "ci_lower", "ci_upper", "p_value"]
    text = "\n\n".join(
        [
            "## Effect of the sign-up promotion (train.csv)",
            markdown_table(result.estimates[cols], 3),
            f"Largest absolute standardized mean difference after weighting: "
            f"{result.balance['smd_weighted'].abs().max():.3f}. Robustness value "
            f"RV = {result.sensitivity['rv']:.4f}.",
        ]
    )
    summary = {
        "estimates": result.estimates[cols].to_dict("records"),
        "overlap": result.overlap,
        "sensitivity": result.sensitivity,
    }
    return text, summary


def targeting_section(cfg: Config) -> tuple[str, dict]:
    """Heterogeneity tests and targeting values on train.csv."""
    train, _ = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    result = analyse_targeting(from_frame(train), "dr", seed=cfg.cv.seed)
    blp = result.blp
    text = "\n\n".join(
        [
            "## Heterogeneity of the promotion effect (train.csv, DR-learner)",
            f"Best linear predictor: ATE {blp['ate']:.3f} (se {blp['ate_se']:.3f}), "
            f"heterogeneity loading {blp['heterogeneity']:.3f} "
            f"(se {blp['heterogeneity_se']:.3f}, "
            f"p = {blp['heterogeneity_p_value']:.3g}).",
            markdown_table(result.gates, 3),
            markdown_table(result.policies, 3),
        ]
    )
    summary = {
        "blp": blp,
        "gates": result.gates_test,
        "policies": result.policies.to_dict("records"),
    }
    return text, summary


def design_section(cfg: Config) -> tuple[str, dict]:
    """Schedule comparison, sensitivities, risk and threshold search for test.csv."""
    train, test = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    result = design_programme(
        train,
        test,
        cfg.rewards.tiers,
        n_simulations=cfg.rewards.n_simulations,
        seed=cfg.rewards.seed,
    )
    risk = result.tail_risk
    text = "\n\n".join(
        [
            f"## Reward programme design ({len(test)} customers in test.csv)",
            markdown_table(result.schedules, 3),
            f"95% value at risk {risk['var']:.2f}, expected shortfall "
            f"{risk['expected_shortfall']:.2f}; expected cost is "
            f"{result.cost_to_spend:.2%} of expected spend.",
            markdown_table(result.sensitivity, 3),
            markdown_table(result.behaviour, 3),
            f"Best schedules with a 95th percentile cost within {result.budget:.2f}:",
            markdown_table(result.search.head(10), 3),
        ]
    )
    summary = {
        "schedules": result.schedules.to_dict("records"),
        "tail_risk": risk,
        "cost_to_spend": result.cost_to_spend,
        "best": result.search.head(10).to_dict("records"),
    }
    return text, summary


def bayes_section(cfg: Config) -> tuple[str, dict]:
    """Hierarchical model with normal and Student-t errors on train.csv."""
    train, _ = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    X, y = split_xy(train)
    rows, contrasts = [], None
    for label, nu in (("normal errors", None), ("Student-t errors, nu = 4", 4.0)):
        post = fit_hierarchical(X, y, nu=nu, seed=cfg.cv.seed)
        diag = post.summary()
        fit = waic(post.log_likelihood(X, y, seed=cfg.cv.seed))
        checks = posterior_predictive_check(post, X, y, seed=cfg.cv.seed)
        rows.append(
            {
                "model": label,
                "waic": fit["waic"],
                "p_waic": fit["p_waic"],
                "max_rhat": float(diag["rhat"].max()),
                "min_ess": float(diag["ess"].min()),
                "min_ppc_p": float(checks["p_value"].min()),
                "max_ppc_p": float(checks["p_value"].max()),
            }
        )
        if contrasts is None:
            contrasts = post.group_contrasts()
    table = pd.DataFrame(rows)
    assert contrasts is not None
    text = "\n\n".join(
        [
            "## Hierarchical Bayesian model (train.csv)",
            markdown_table(table, 3),
            "Region effects relative to their mean (normal errors):",
            markdown_table(contrasts, 4),
        ]
    )
    return text, {"models": rows, "region_contrasts": contrasts.to_dict("records")}


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
        + markdown_table(table, 2)
        + f"\n\nCustomers with at least a 20% chance of changing tier: {near}."
    )


def build_report(
    cfg: Config | None = None,
    settings: ReportSettings | None = None,
    sections: Sequence[str] = (
        "drivers",
        "lineage",
        "benchmark",
        "study",
        "conformal",
        "probabilistic",
        "promotion",
        "targeting",
        "design",
        "bayes",
    ),
) -> Path:
    """Run the analysis and write ``reports/results.md`` with figures."""
    cfg = cfg or load_config()
    settings = settings or ReportSettings()
    report_dir = Path(cfg.paths.report_dir)
    fig_dir = report_dir / "figures"
    parts = ["# Results", "Generated by `loyalty-spend report`."]
    summary: dict = {}
    if "drivers" in sections:
        text, summary["drivers"] = driver_section(cfg, fig_dir)
        parts.append(text)
    if "lineage" in sections:
        parts.append(lineage_section(cfg))
    residuals = None
    if "benchmark" in sections:
        text, table, residuals = benchmark_section(cfg, settings, fig_dir)
        parts.append(text)
        summary["benchmark"] = table.to_dict("records")
        parts.append(liability_section(cfg, residuals, fig_dir))
    if "study" in sections:
        parts.append(study_section(cfg, settings, fig_dir))
    if "conformal" in sections:
        parts.append(conformal_section(cfg, settings, fig_dir))
    if "promotion" in sections:
        text, summary["promotion"] = promotion_section(cfg, fig_dir)
        parts.append(text)
    if "targeting" in sections:
        text, summary["targeting"] = targeting_section(cfg)
        parts.append(text)
    if "design" in sections:
        text, summary["design"] = design_section(cfg)
        parts.append(text)
    if "bayes" in sections:
        text, summary["bayes"] = bayes_section(cfg)
        parts.append(text)
    if "probabilistic" in sections:
        text, summary["probabilistic"] = probabilistic_section(cfg, settings, fig_dir)
        parts.append(text)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "results.md"
    path.write_text("\n\n".join(parts) + "\n")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return path
