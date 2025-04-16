"""Command line interface: ``loyalty-spend <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

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


@command("rewards", "forecast the reward cost of a customer file", _rewards_args)
def _rewards(args: argparse.Namespace) -> int:
    from loyalty_spend.business.rewards import RewardSchedule, forecast_liability
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.data.schema import ID
    from loyalty_spend.models.persistence import load_bundle

    cfg = load_config(args.config)
    bundle = load_bundle(args.model)
    frame, _ = load_clean(args.input, impute=False, require_target=False)
    fc = forecast_liability(
        frame[ID],
        bundle.predict(frame),
        bundle.residuals,
        RewardSchedule(cfg.rewards.tiers),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loyalty-spend")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, arguments, handler) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        arguments(p)
        p.set_defaults(func=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
