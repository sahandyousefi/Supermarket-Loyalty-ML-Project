"""Command line interface: ``loyalty-spend <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from loyalty_spend import __version__
from loyalty_spend.config import load_config

Handler = Callable[[argparse.Namespace], int]
_COMMANDS: dict[
    str, tuple[str, Callable[[argparse.ArgumentParser], None], Handler]
] = {}


def command(name: str, help: str, arguments: Callable[[argparse.ArgumentParser], None]):
    """Register a subcommand with its argument builder."""

    def decorator(handler: Handler) -> Handler:
        _COMMANDS[name] = (help, arguments, handler)
        return handler

    return decorator


def _config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML configuration file")


def _clean_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--input", default=None, help="raw CSV, defaults to loyalty.csv")
    p.add_argument("--output", required=True)


@command("clean", "clean a raw extract to the data dictionary", _clean_args)
def _clean(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    path = args.input or cfg.paths.raw(cfg.paths.loyalty_file)
    clean, report = load_clean(path)
    clean.to_csv(args.output, index=False)
    print(report.to_frame().to_string(index=False))
    print(f"wrote {len(clean)} rows to {args.output}")
    return 0


def _validate_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--input", default=None, help="raw CSV, defaults to train.csv")
    p.add_argument(
        "--contract",
        default=None,
        help="contract name; defaults to training, or scoring without a spend column",
    )
    p.add_argument("--contracts", default=None, help="YAML file with contracts")
    p.add_argument("--output", default=None, help="write violations to this CSV")


@command("validate", "check an extract against a data contract", _validate_args)
def _validate(args: argparse.Namespace) -> int:
    from loyalty_spend.data.cleaning import clean_loyalty
    from loyalty_spend.data.contracts import check_contract, load_contracts
    from loyalty_spend.data.io import read_raw
    from loyalty_spend.data.schema import TARGET

    cfg = load_config(args.config)
    path = args.input or cfg.paths.raw(cfg.paths.train_file)
    raw = read_raw(path)
    has_target = TARGET in raw.columns
    clean, _ = clean_loyalty(raw, impute=False, require_target=has_target)
    contracts = load_contracts(args.contracts)
    name = args.contract or ("training" if has_target else "scoring")
    report = check_contract(clean, contracts[name])
    table = report.to_frame()
    if args.output:
        table.to_csv(args.output, index=False)
    status = "passed" if report.ok else "failed"
    print(f"{path}: {report.n_rows} rows, contract '{name}' {status}")
    if not table.empty:
        print(table.to_string(index=False))
    return 0 if report.ok else 1


def _drift_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--reference", default=None, help="defaults to train.csv")
    p.add_argument("--current", default=None, help="defaults to test.csv")
    p.add_argument("--output", default=None, help="write the report to this CSV")
    p.add_argument(
        "--fail-on",
        choices=["never", "moderate", "large"],
        default="never",
        help="exit with status 1 when any feature reaches this level",
    )


@command("drift", "compare a new extract with the reference data", _drift_args)
def _drift(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.monitoring.domain import domain_classifier_test
    from loyalty_spend.monitoring.drift import DriftThresholds, drift_report

    cfg = load_config(args.config)
    ref_path = args.reference or cfg.paths.raw(cfg.paths.train_file)
    cur_path = args.current or cfg.paths.raw(cfg.paths.test_file)
    reference, _ = load_clean(ref_path, impute=False, require_target=False)
    current, _ = load_clean(cur_path, impute=False, require_target=False)
    report = drift_report(
        reference,
        current,
        n_bins=cfg.monitoring.n_bins,
        thresholds=DriftThresholds.from_config(cfg.monitoring),
    )
    domain = domain_classifier_test(reference, current, seed=cfg.cv.seed)
    if args.output:
        report.to_csv(args.output, index=False)
    cols = ["feature", "psi", "statistic", "p_adjusted", "unseen", "status"]
    print(report[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    alert = domain.auc >= cfg.monitoring.domain_auc_alert
    print(
        f"domain classifier AUC {domain.auc:.3f} (p = {domain.p_value:.3g})"
        + (" exceeds the alert level" if alert else "")
    )
    levels = {"never": set(), "moderate": {"moderate", "large"}, "large": {"large"}}
    failed = bool(set(report["status"]) & levels[args.fail_on])
    return 1 if failed else 0


def _intervals_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument(
        "--scenario",
        default="noisy",
        help="simulation scenario calibrated to train.csv",
    )
    p.add_argument("--n-fit", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="write the table to this CSV")


@command("intervals", "compare prediction interval methods", _intervals_args)
def _intervals(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_modeling_data
    from loyalty_spend.data.schema import FEATURES, TARGET
    from loyalty_spend.data.simulate import SCENARIOS, simulate
    from loyalty_spend.uncertainty.compare import compare_intervals

    cfg = load_config(args.config)
    if args.scenario not in SCENARIOS:
        raise SystemExit(f"unknown scenario; choose from {sorted(SCENARIOS)}")
    train, _ = load_modeling_data(
        cfg.paths.raw(cfg.paths.train_file), cfg.paths.raw(cfg.paths.test_file)
    )
    sim = simulate(
        train, SCENARIOS[args.scenario], n=args.n_fit + args.n_test, seed=args.seed
    ).frame
    fit, new = sim.iloc[: args.n_fit], sim.iloc[args.n_fit :]
    table = compare_intervals(
        fit[FEATURES],
        fit[TARGET],
        new[FEATURES],
        new[TARGET],
        alpha=cfg.conformal.alpha,
        calibration_size=cfg.conformal.calibration_size,
        seed=args.seed,
    )
    if args.output:
        table.to_csv(args.output, index=False)
    cols = [
        "method",
        "coverage",
        "mean_width",
        "interval_score",
        "worst_group_coverage",
        "best_group_coverage",
    ]
    print(f"scenario '{args.scenario}', target coverage {1 - cfg.conformal.alpha:.0%}")
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def _effect_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument(
        "--input", default=None, help="CSV with spend, defaults to train.csv"
    )
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--output-dir", default=None, help="write tables to this folder")


@command("effect", "estimate the effect of the sign-up promotion", _effect_args)
def _effect(args: argparse.Namespace) -> int:
    from loyalty_spend.causal.analysis import analyse_promotion
    from loyalty_spend.causal.data import from_frame
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    path = args.input or cfg.paths.raw(cfg.paths.train_file)
    frame, _ = load_clean(path, impute=False)
    data = from_frame(frame)
    result = analyse_promotion(data, n_boot=args.n_boot, seed=cfg.cv.seed)
    cols = ["method", "estimand", "estimate", "se", "ci_lower", "ci_upper", "p_value"]
    print(f"{path}: {data.n_treated} treated, {data.n_control} control")
    print(
        result.estimates[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    ov = result.overlap
    print(
        f"propensity range {min(ov['treated_min'], ov['control_min']):.3f} to "
        f"{max(ov['treated_max'], ov['control_max']):.3f}; "
        f"largest |SMD| after weighting "
        f"{result.balance['smd_weighted'].abs().max():.3f}"
    )
    sens = result.sensitivity
    print(f"robustness value {sens['rv']:.4f} (RV at 5%: {sens['rv_alpha']:.4f})")
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.estimates.to_csv(out / "effect_estimates.csv", index=False)
        result.balance.to_csv(out / "balance.csv", index=False)
        (out / "effect_diagnostics.json").write_text(
            json.dumps({"overlap": ov, "sensitivity": sens}, indent=2)
        )
    return 0


def _uplift_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument(
        "--input", default=None, help="CSV with spend, defaults to train.csv"
    )
    p.add_argument("--learner", choices=["s", "t", "x", "dr", "r"], default="dr")
    p.add_argument("--cost", type=float, default=0.0, help="cost of one promotion")
    p.add_argument("--margin", type=float, default=1.0, help="profit per unit of spend")
    p.add_argument("--output-dir", default=None, help="write tables to this folder")


@command("uplift", "test effect heterogeneity and value targeting rules", _uplift_args)
def _uplift(args: argparse.Namespace) -> int:
    from loyalty_spend.causal.data import from_frame
    from loyalty_spend.causal.targeting import analyse_targeting
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    path = args.input or cfg.paths.raw(cfg.paths.train_file)
    frame, _ = load_clean(path, impute=False)
    result = analyse_targeting(
        from_frame(frame), args.learner, args.cost, args.margin, seed=cfg.cv.seed
    )
    blp = result.blp
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(
        f"BLP: ATE {blp['ate']:.3f} (se {blp['ate_se']:.3f}), heterogeneity "
        f"loading {blp['heterogeneity']:.3f} (p = {blp['heterogeneity_p_value']:.3g})"
    )
    print(result.gates.to_string(index=False, float_format=fmt))
    print(result.policies.to_string(index=False, float_format=fmt))
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.gates.to_csv(out / "gates.csv", index=False)
        result.profile.to_csv(out / "profile.csv", index=False)
        result.policies.to_csv(out / "policies.csv", index=False)
        result.budget.to_csv(out / "budget_curve.csv", index=False)
        (out / "uplift_summary.json").write_text(
            json.dumps(
                {"blp": blp, "gates": result.gates_test, "qini": result.qini}, indent=2
            )
        )
    return 0


def _drivers_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--group", default="loyalty_years")


@command("drivers", "test whether a feature drives spend", _drivers_args)
def _drivers(args: argparse.Namespace) -> int:
    from loyalty_spend.analysis.drivers import effect_sizes, spend_by_group, welch_anova
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    clean, _ = load_clean(cfg.paths.raw(cfg.paths.loyalty_file), impute=False)
    print(spend_by_group(clean, args.group).round(2).to_string(index=False))
    print(
        json.dumps(
            {**welch_anova(clean, args.group), **effect_sizes(clean, args.group)}
        )
    )
    return 0


def _compare_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--models", nargs="+", default=["ols", "ridge", "hist_gbm"])
    p.add_argument("--reference", default="ols")
    p.add_argument("--output", default=None, help="CSV of the comparison table")


@command("compare", "repeated cross-validation of several models", _compare_args)
def _compare(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.evaluation.cv import compare_results, cross_validate, make_splits
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.registry import build_model

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    X, y = split_xy(train)
    splits = make_splits(len(y), cfg.cv.n_splits, cfg.cv.n_repeats, cfg.cv.seed)
    results = [cross_validate(m, build_model(m), X, y, splits) for m in args.models]
    table = compare_results(results, args.reference)
    if args.output:
        table.to_csv(args.output, index=False)
    print(table.round(4).to_string(index=False))
    return 0


def _train_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--model", default="ols")
    p.add_argument(
        "--output", default=None, help="defaults to <output_dir>/model.joblib"
    )


@command("train", "fit a model and save it with its residuals", _train_args)
def _train(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.evaluation.cv import cross_validate, make_splits
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.persistence import (
        ModelBundle,
        data_fingerprint,
        save_bundle,
    )
    from loyalty_spend.models.registry import build_model

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    X, y = split_xy(train)
    splits = make_splits(len(y), cfg.cv.n_splits, 1, cfg.cv.seed)
    cv = cross_validate(args.model, build_model(args.model), X, y, splits)
    model = build_model(args.model).fit(X, y)
    bundle = ModelBundle(
        args.model,
        model,
        residuals=y.to_numpy() - cv.oof,
        metadata={"cv": cv.summary(), "data": data_fingerprint(train)},
    )
    path = save_bundle(
        bundle, args.output or Path(cfg.paths.output_dir) / "model.joblib"
    )
    print(json.dumps(cv.summary(), indent=2))
    print(f"saved {path}")
    return 0


def _predict_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="CSV of customers to score")
    p.add_argument("output")
    p.add_argument("--model", default="artifacts/model.joblib")


@command("predict", "score customers with a saved model", _predict_args)
def _predict(args: argparse.Namespace) -> int:
    import pandas as pd

    from loyalty_spend.data.io import load_clean
    from loyalty_spend.data.schema import ID, TARGET
    from loyalty_spend.models.persistence import load_bundle

    bundle = load_bundle(args.model)
    frame, _ = load_clean(args.input, impute=False, require_target=False)
    pred = bundle.predict(frame)
    out = pd.DataFrame({ID: frame[ID], TARGET: pred.round(2)})
    out.to_csv(args.output, index=False)
    print(f"scored {len(out)} customers with '{bundle.name}', wrote {args.output}")
    return 0


def _rewards_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("input", help="CSV of customers to cost")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--output", default=None, help="per-customer CSV")
    p.add_argument(
        "--uncertainty",
        choices=["residual", "heteroscedastic"],
        default="residual",
        help="residual bootstrap around the saved model, or a heteroscedastic "
        "Gaussian model fitted to train.csv",
    )


@command("rewards", "forecast the reward cost of a customer file", _rewards_args)
def _rewards(args: argparse.Namespace) -> int:
    from loyalty_spend.business.rewards import (
        RewardSchedule,
        forecast_liability,
        gaussian_sampler,
        simulate_liability,
    )
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.data.schema import ID
    from loyalty_spend.features.preprocess import feature_frame, split_xy

    cfg = load_config(args.config)
    frame, _ = load_clean(args.input, impute=False, require_target=False)
    schedule = RewardSchedule(cfg.rewards.tiers)
    if args.uncertainty == "heteroscedastic":
        from loyalty_spend.models.distributional import (
            HeteroscedasticGaussianRegressor,
        )

        train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
        model = HeteroscedasticGaussianRegressor().fit(*split_xy(train))
        X = feature_frame(frame)
        mean, sigma = model.predict(X), model.predict_sigma(X)
        fc = simulate_liability(
            frame[ID],
            mean,
            gaussian_sampler(mean, sigma),
            schedule,
            cfg.rewards.n_simulations,
            cfg.rewards.seed,
        )
    else:
        from loyalty_spend.models.persistence import load_bundle

        bundle = load_bundle(args.model)
        fc = forecast_liability(
            frame[ID],
            bundle.predict(frame),
            bundle.residuals,
            schedule,
            cfg.rewards.n_simulations,
            cfg.rewards.seed,
        )
    if args.output:
        fc.per_customer.to_csv(args.output, index=False)
    print(json.dumps({k: round(v, 2) for k, v in fc.summary().items()}, indent=2))
    return 0


def _tune_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--model", default="lightgbm")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--nested", action="store_true", help="nested CV estimate")


@command("tune", "Optuna search with optional nested CV", _tune_args)
def _tune(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.tuning import nested_cv, tune

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    X, y = split_xy(train)
    if args.nested:
        res = nested_cv(args.model, X, y, n_trials=args.trials, seed=cfg.cv.seed)
        print(json.dumps({"outer_rmse": res.mean, "optimism": res.optimism}))
    else:
        params, score = tune(
            args.model, X, y, args.trials, cfg.cv.n_splits, cfg.cv.seed
        )
        print(json.dumps({"cv_rmse": score, "params": params}, default=str))
    return 0


@command("tasks", "write the four deliverables of the original brief", _config_arg)
def _tasks(args: argparse.Namespace) -> int:
    from loyalty_spend.tasks import run_all

    results = run_all(load_config(args.config))
    for name, frame in results.items():
        print(f"{name}: {len(frame)} rows")
    return 0


REPORT_SECTIONS = (
    "drivers",
    "lineage",
    "benchmark",
    "study",
    "conformal",
    "probabilistic",
    "promotion",
    "targeting",
)


def _report_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument(
        "--sections",
        nargs="+",
        choices=REPORT_SECTIONS,
        default=list(REPORT_SECTIONS),
    )


@command("report", "run the full analysis and write reports/results.md", _report_args)
def _report(args: argparse.Namespace) -> int:
    from loyalty_spend.reporting.report import build_report

    path = build_report(load_config(args.config), sections=args.sections)
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loyalty-spend")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="log progress (-v) or debugging detail (-vv) to stderr",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, arguments, handler) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        arguments(p)
        p.set_defaults(func=handler)
    return parser


def configure_logging(verbosity: int) -> None:
    """Warnings only by default; ``-v`` adds progress, ``-vv`` debug output."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
