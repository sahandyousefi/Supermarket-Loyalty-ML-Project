"""Command line interface: ``loyalty-spend <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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


def _segments_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument(
        "--input", default=None, help="CSV to segment, defaults to loyalty.csv"
    )
    p.add_argument(
        "--k", type=int, default=None, help="number of segments; chosen if omitted"
    )
    p.add_argument("--max-k", type=int, default=8)
    p.add_argument(
        "--n-boot", type=int, default=30, help="bootstrap resamples for stability"
    )
    p.add_argument("--output-dir", default=None, help="write tables to this folder")


@command("segments", "segment members on their recorded attributes", _segments_args)
def _segments(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean, read_raw
    from loyalty_spend.data.schema import FEATURES, ID, TARGET
    from loyalty_spend.segments.gower import gower_matrix
    from loyalty_spend.segments.kmedoids import KMedoids
    from loyalty_spend.segments.profiles import profile_segments
    from loyalty_spend.segments.selection import choose_k, silhouette_table
    from loyalty_spend.segments.stability import bootstrap_stability

    cfg = load_config(args.config)
    path = args.input or cfg.paths.raw(cfg.paths.loyalty_file)
    has_spend = TARGET in read_raw(path).columns
    frame, _ = load_clean(path, impute=False, require_target=has_spend)
    X = frame[FEATURES].reset_index(drop=True)
    D = gower_matrix(X)
    table = silhouette_table(D, range(2, args.max_k + 1), seed=cfg.cv.seed)
    k = args.k or choose_k(table, min_size=max(10, len(X) // 50))
    model = KMedoids(k, seed=cfg.cv.seed).fit(D)
    spend = frame.get(TARGET)
    profiles = profile_segments(X, model.labels_, spend)
    stability = bootstrap_stability(
        D, model.labels_, n_boot=args.n_boot, seed=cfg.cv.seed
    )
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(table.to_string(index=False, float_format=fmt))
    print(f"\n{k} segments")
    print(profiles.to_string(index=False, float_format=fmt))
    print(stability.to_string(index=False, float_format=fmt))
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "silhouette.csv", index=False)
        profiles.to_csv(out / "profiles.csv", index=False)
        stability.to_csv(out / "stability.csv", index=False)
        frame[[ID]].assign(segment=model.labels_).to_csv(
            out / "assignments.csv", index=False
        )
    return 0


def _design_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument(
        "--input", default=None, help="customers to cost, defaults to test.csv"
    )
    p.add_argument(
        "--budget", type=float, default=None, help="cap on the 95th percentile"
    )
    p.add_argument("--offsets", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    p.add_argument("--output-dir", default=None, help="write tables to this folder")


@command("design", "compare reward schedules and search thresholds", _design_args)
def _design(args: argparse.Namespace) -> int:
    from loyalty_spend.business.programme import design_programme
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    path = args.input or cfg.paths.raw(cfg.paths.test_file)
    customers, _ = load_clean(path, impute=False, require_target=False)
    result = design_programme(
        train,
        customers,
        cfg.rewards.tiers,
        budget=args.budget,
        offsets=tuple(args.offsets),
        n_simulations=cfg.rewards.n_simulations,
        seed=cfg.rewards.seed,
    )
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(result.schedules.to_string(index=False, float_format=fmt))
    risk = result.tail_risk
    print(
        f"95% value at risk {risk['var']:.2f}, expected shortfall "
        f"{risk['expected_shortfall']:.2f}; cost is {result.cost_to_spend:.2%} of spend"
    )
    print(result.sensitivity.to_string(index=False, float_format=fmt))
    print(f"best schedules within a budget of {result.budget:.2f}:")
    print(result.search.head(5).to_string(index=False, float_format=fmt))
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.schedules.to_csv(out / "schedules.csv", index=False)
        result.sensitivity.to_csv(out / "sensitivity.csv", index=False)
        result.behaviour.to_csv(out / "behaviour.csv", index=False)
        result.search.to_csv(out / "threshold_search.csv", index=False)
        (out / "risk.json").write_text(
            json.dumps({**risk, "cost_to_spend": result.cost_to_spend}, indent=2)
        )
    return 0


def _monitor_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("batch", help="CSV of customers to be scored")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--output", default=None, help="write the result as JSON")


@command("monitor", "check a scoring batch before it is scored", _monitor_args)
def _monitor(args: argparse.Namespace) -> int:
    from loyalty_spend.data.contracts import load_contracts
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.models.persistence import load_bundle
    from loyalty_spend.ops.monitor import monitor_batch

    cfg = load_config(args.config)
    reference, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    batch, _ = load_clean(args.batch, impute=False, require_target=False)
    result = monitor_batch(
        reference,
        batch,
        load_bundle(args.model),
        cfg.monitoring,
        load_contracts()["scoring"],
        seed=cfg.cv.seed,
    )
    print(f"status: {result.status}")
    for reason in result.reasons:
        print(f"- {reason}")
    if args.output:
        Path(args.output).write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )
    return {"ok": 0, "warn": 0, "alert": 2}[result.status]


def _serve_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)


@command("serve", "run the HTTP scoring service", _serve_args)
def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("install the serve extra: pip install '.[serve]'") from exc
    from loyalty_spend.serving.api import create_app

    uvicorn.run(create_app(args.model, args.config), host=args.host, port=args.port)
    return 0


def _card_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--output", default="artifacts/MODEL_CARD.md")


@command("model-card", "write a model card for a saved model", _card_args)
def _card(args: argparse.Namespace) -> int:
    from loyalty_spend.models.persistence import load_bundle
    from loyalty_spend.ops.model_card import write_model_card

    path = write_model_card(load_bundle(args.model), args.output)
    print(f"wrote {path}")
    return 0


def _narrate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--summary", default="reports/summary.json")
    p.add_argument(
        "--offline",
        action="store_true",
        help="use the fact template even if a language model is configured",
    )


@command("narrate", "summarise the report in plain language", _narrate_args)
def _narrate(args: argparse.Namespace) -> int:
    from loyalty_spend.llm.client import client_from_env
    from loyalty_spend.llm.facts import build_fact_sheet
    from loyalty_spend.llm.narrative import generate_narrative, template_narrative
    from loyalty_spend.llm.usage import TrackedClient

    facts = build_fact_sheet(json.loads(Path(args.summary).read_text()))
    client = None if args.offline else client_from_env()
    if client is None:
        narrative = template_narrative(facts)
    else:
        tracked = TrackedClient(client)
        narrative = generate_narrative(tracked, facts)
    print(narrative.text)
    if client is not None:
        usage = tracked.summary()
        print(
            f"\n{narrative.model}, prompt v{narrative.prompt_version}: "
            f"{usage['calls']:.0f} calls, {usage['prompt_tokens']:.0f} prompt and "
            f"{usage['completion_tokens']:.0f} completion tokens"
        )
    return 0


def _ask_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("question")
    p.add_argument("-k", type=int, default=4, help="passages shown to the model")
    p.add_argument("--offline", action="store_true", help="show passages only")


@command("ask", "answer a question from the project documentation", _ask_args)
def _ask(args: argparse.Namespace) -> int:
    from loyalty_spend.llm.client import client_from_env
    from loyalty_spend.llm.corpus import load_corpus
    from loyalty_spend.llm.guardrails import check_question
    from loyalty_spend.llm.qa import BLOCKED, answer_question
    from loyalty_spend.llm.retrieval import Bm25Retriever

    retriever = Bm25Retriever(load_corpus())
    client = None if args.offline else client_from_env()
    if client is None:
        guard = check_question(args.question)
        if not guard.allowed:
            print(BLOCKED)
            return 1
        for i, hit in enumerate(retriever.search(guard.text, args.k), start=1):
            print(f"[{i}] {hit.chunk.citation} (score {hit.score:.3f})")
            print(hit.chunk.text[:400].strip() + "\n")
        return 0
    answer = answer_question(client, retriever, args.question, args.k)
    print(answer.answer)
    for i, chunk in enumerate(answer.sources, start=1):
        print(f"  [{i}] {chunk.citation}")
    if not answer.refused and not answer.verified:
        print("note: some sentences are not fully supported by their citations")
    return 1 if answer.answer == BLOCKED else 0


def _bayes_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--group", default="region", help="partially pooled factor")
    p.add_argument(
        "--nu", type=float, default=None, help="Student-t degrees of freedom"
    )
    p.add_argument("--variance-by", default=None, help="column with its own variance")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--iter", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--output-dir", default=None, help="write tables to this folder")


@command("bayes", "fit the hierarchical Bayesian model of spend", _bayes_args)
def _bayes(args: argparse.Namespace) -> int:
    from loyalty_spend.bayes.checks import posterior_predictive_check
    from loyalty_spend.bayes.compare import waic
    from loyalty_spend.bayes.hierarchical import fit_hierarchical, posterior_sampler
    from loyalty_spend.business.rewards import RewardSchedule, simulate_liability
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.data.schema import ID
    from loyalty_spend.features.preprocess import split_xy

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    test, _ = load_clean(
        cfg.paths.raw(cfg.paths.test_file), impute=False, require_target=False
    )
    X, y = split_xy(train)
    post = fit_hierarchical(
        X,
        y,
        group=args.group,
        nu=args.nu,
        variance_by=args.variance_by,
        n_chains=args.chains,
        n_iter=args.iter,
        n_warmup=args.warmup,
        seed=cfg.cv.seed,
    )
    summary = post.summary()
    checks = posterior_predictive_check(post, X, y, seed=cfg.cv.seed)
    fit = waic(post.log_likelihood(X, y, seed=cfg.cv.seed))
    cost = simulate_liability(
        test[ID],
        post.predict(test),
        posterior_sampler(post, test),
        RewardSchedule(cfg.rewards.tiers),
        cfg.rewards.n_simulations,
        cfg.rewards.seed,
    )
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    keep = summary["parameter"].str.match(r"^(u\[|sigma|tau)")
    print(summary[keep].to_string(index=False, float_format=fmt))
    print(f"max R-hat {summary['rhat'].max():.3f}, min ESS {summary['ess'].min():.0f}")
    print(checks.to_string(index=False, float_format=fmt))
    print(f"WAIC {fit['waic']:.1f} (p_waic {fit['p_waic']:.1f})")
    s = cost.summary()
    print(
        f"reward cost for test.csv: mean {s['expected_total']:.2f}, "
        f"90% interval {s['p05']:.2f} to {s['p95']:.2f}"
    )
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "posterior_summary.csv", index=False)
        checks.to_csv(out / "posterior_checks.csv", index=False)
        (out / "bayes.json").write_text(json.dumps({"waic": fit, "cost": s}, indent=2))
    return 0


def _next_tier_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("input", help="CSV of customers")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--output", default=None)


@command(
    "next-tier",
    "first-month spend each customer needs for the next tier",
    _next_tier_args,
)
def _next_tier(args: argparse.Namespace) -> int:
    from loyalty_spend.business.rewards import RewardSchedule
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.explain.counterfactual import next_tier_table
    from loyalty_spend.models.persistence import load_bundle

    cfg = load_config(args.config)
    bundle = load_bundle(args.model)
    reference, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    customers, _ = load_clean(args.input, impute=False, require_target=False)
    customers = customers.dropna(subset=["first_month"]).reset_index(drop=True)
    table = next_tier_table(
        bundle.predict, customers, RewardSchedule(cfg.rewards.tiers), reference
    )
    if args.output:
        table.to_csv(args.output, index=False)
    below_top = table["next_threshold"].notna()
    print(
        f"{int(below_top.sum())} of {len(table)} customers are below the top tier; "
        f"{int(table.loc[below_top, 'feasible'].sum())} could reach the next tier "
        "with a plausible first-month spend"
    )
    return 0


def _robustness_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--model", default="ols", help="registry model to examine")
    p.add_argument("--output-dir", default=None, help="write tables to this folder")


@command(
    "robustness",
    "subgroup errors, slices, stress tests and influence",
    _robustness_args,
)
def _robustness(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.data.schema import CATEGORICAL
    from loyalty_spend.evaluation.cv import cross_validate, make_splits
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.registry import build_model
    from loyalty_spend.robustness.influence import influence_table
    from loyalty_spend.robustness.learning_curve import learning_curve
    from loyalty_spend.robustness.mnar import delta_sensitivity
    from loyalty_spend.robustness.slices import find_slices
    from loyalty_spend.robustness.stress import stress_test
    from loyalty_spend.robustness.subgroups import subgroup_errors

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    X, y = split_xy(train)
    splits = make_splits(len(y), cfg.cv.n_splits, 1, cfg.cv.seed)
    cv = cross_validate(args.model, build_model(args.model), X, y, splits)
    groups = subgroup_errors(y, cv.oof, X, list(CATEGORICAL), seed=cfg.cv.seed)
    slices = find_slices(y, cv.oof, X, list(X.columns))
    cut = int(0.7 * len(y))
    fitted = build_model(args.model).fit(X.iloc[:cut], y.iloc[:cut])
    stress = stress_test(
        fitted.predict, X.iloc[cut:].reset_index(drop=True), y.iloc[cut:]
    )
    curve = learning_curve(build_model(args.model), X, y, splits, seed=cfg.cv.seed)
    full = build_model(args.model).fit(X, y)
    tables = {
        "subgroups": groups,
        "slices": slices,
        "stress": stress,
        "learning_curve": curve,
    }
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    worst = groups.sort_values("rmse_ratio").tail(3)
    print("largest subgroup errors:")
    print(
        worst[["attribute", "level", "n", "rmse", "rmse_ratio"]].to_string(
            index=False, float_format=fmt
        )
    )
    print(
        f"slices flagged after FDR control: {int(slices['flagged'].sum())} "
        f"of {len(slices)}"
    )
    print(
        stress.pivot(
            index="scenario", columns="severity", values="rmse_ratio"
        ).to_string(float_format=fmt)
    )
    print(
        curve[["n_train", "rmse", "lower", "upper"]].to_string(
            index=False, float_format=fmt
        )
    )
    if hasattr(full, "steps"):
        infl = influence_table(full, X, y)
        tables["influence"] = infl
        print(
            f"influential members: {int(infl['flag_cooks'].sum())} by Cook's distance, "
            f"largest D = {infl['cooks_distance'].max():.3f}"
        )
    loyalty, _ = load_clean(
        cfg.paths.raw(cfg.paths.loyalty_file), impute=False, require_target=False
    )
    mnar = delta_sensitivity(
        loyalty.drop(columns=["customer_id"]),
        lambda d: float(full.predict(d).mean()),
        seed=cfg.cv.seed,
    )
    tables["mnar"] = mnar
    print(mnar.to_string(index=False, float_format=fmt))
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=name == "influence")
    return 0


def _synthesize_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--input", default=None, help="cleaned extract to imitate")
    p.add_argument("--output", default="synthetic.csv")
    p.add_argument("--rows", type=int, default=None, help="rows to generate")
    p.add_argument(
        "--copula",
        choices=["plain", "stratified", "conditional"],
        default="conditional",
        help="synthesiser to use",
    )
    p.add_argument("--by", default="loyalty_years", help="stratifying attribute")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--evaluate", action="store_true", help="report fidelity, utility, privacy"
    )


def _make_synthesizer(kind: str, by: str, seed: int) -> Any:
    from loyalty_spend.synthetic.conditional import ConditionalCopulaSynthesizer
    from loyalty_spend.synthetic.copula import (
        GaussianCopulaSynthesizer,
        StratifiedCopulaSynthesizer,
    )

    if kind == "plain":
        return GaussianCopulaSynthesizer(seed=seed)
    if kind == "stratified":
        return StratifiedCopulaSynthesizer(by, seed=seed)
    return ConditionalCopulaSynthesizer(by, seed=seed)


@command("synthesize", "generate synthetic members with a copula", _synthesize_args)
def _synthesize(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean

    cfg = load_config(args.config)
    path = args.input or cfg.paths.raw(cfg.paths.train_file)
    real, _ = load_clean(path, impute=False)
    data = real.drop(columns="customer_id")
    n = args.rows or len(data)
    synth = _make_synthesizer(args.copula, args.by, args.seed).fit(data).sample(n)
    synth.insert(0, "customer_id", [f"S{i:05d}" for i in range(1, n + 1)])
    synth.to_csv(args.output, index=False)
    print(f"{n} synthetic members written to {args.output}")
    if not args.evaluate:
        return 0
    from loyalty_spend.evaluation.cv import make_splits
    from loyalty_spend.synthetic.fidelity import fidelity_summary
    from loyalty_spend.synthetic.membership import membership_inference
    from loyalty_spend.synthetic.privacy import privacy_report
    from loyalty_spend.synthetic.utility import (
        train_synthetic_test_real,
        utility_summary,
    )

    fid = fidelity_summary(data, synth.drop(columns="customer_id"))
    splits = make_splits(len(data), cfg.cv.n_splits, 1, cfg.cv.seed)
    util = utility_summary(
        train_synthetic_test_real(
            data,
            {args.copula: lambda k: _make_synthesizer(args.copula, args.by, k)},
            splits,
        )
    )
    # Privacy is judged on a synthesiser that saw only half of the members.
    shuffled = data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    half = len(shuffled) // 2
    fit_part, held = shuffled.iloc[:half], shuffled.iloc[half:]
    released = (
        _make_synthesizer(args.copula, args.by, args.seed)
        .fit(fit_part)
        .sample(len(fit_part))
    )
    priv = privacy_report(fit_part, held, released)
    mia = membership_inference(fit_part, held, released, seed=args.seed)
    print("fidelity:", ", ".join(f"{k} {v:.3f}" for k, v in fid.items()))
    print(util.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("privacy:", ", ".join(f"{k} {v:.4f}" for k, v in priv.items()))
    print(
        f"membership inference AUC {mia['auc']:.3f} "
        f"[{mia['auc_lower']:.3f}, {mia['auc_upper']:.3f}], "
        f"advantage {mia['advantage']:.3f}"
    )
    return 0


def _control_chart_args(p: argparse.ArgumentParser) -> None:
    _config_arg(p)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--shift", type=float, default=0.0, help="drop in spend")
    p.add_argument("--band", default="1-3", help="tenure band that changes")
    p.add_argument("--start", type=int, default=100, help="first changed batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="CSV of the batch statistics")


@command(
    "control-chart",
    "EWMA, CUSUM, HWMA and MHWMA charts on a simulated stream of batches",
    _control_chart_args,
)
def _control_chart(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.evaluation.cv import cross_validate, make_splits
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.registry import build_model
    from loyalty_spend.monitoring.residual_stream import band_stream, chart_stream

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    X, y = split_xy(train)
    splits = make_splits(len(y), cfg.cv.n_splits, 1, cfg.cv.seed)
    cv = cross_validate("ols", build_model("ols"), X, y, splits)
    stream = band_stream(
        y.to_numpy() - cv.oof,
        X["loyalty_years"].astype(str),
        args.batch_size,
        args.batches,
        args.shift,
        args.band if args.shift else None,
        args.start,
        args.seed,
    )
    table = chart_stream(stream)
    if args.shift:
        table["delay"] = table["first_signal"] - args.start
        print(
            f"spend of band {args.band} lowered by {args.shift:g} "
            f"from batch {args.start}"
        )
    print(table.to_string(index=False))
    # Run lengths are close to geometric, so false alarms are expected too.
    p_false = 1 - (1 - 1 / 370) ** args.batches
    print(
        f"each chart is set for an in-control run length of 370 batches, so it "
        f"raises a false alarm within {args.batches} batches with probability "
        f"about {p_false:.2f}"
    )
    if args.output:
        stream.to_csv(args.output, index_label="batch")
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
    p.add_argument(
        "--jobs", type=int, default=1, help="processes for fold fitting (-1: all)"
    )


@command("compare", "repeated cross-validation of several models", _compare_args)
def _compare(args: argparse.Namespace) -> int:
    from loyalty_spend.data.io import load_clean
    from loyalty_spend.evaluation.cv import compare_results, cross_validate, make_splits
    from loyalty_spend.evaluation.fold_cache import FoldCache
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.registry import build_model

    cfg = load_config(args.config)
    train, _ = load_clean(cfg.paths.raw(cfg.paths.train_file), impute=False)
    X, y = split_xy(train)
    splits = make_splits(len(y), cfg.cv.n_splits, cfg.cv.n_repeats, cfg.cv.seed)
    # Models with the same encoding share the preprocessing of each fold.
    cache = FoldCache(X, y, splits) if args.jobs == 1 else None
    results = [
        cross_validate(m, build_model(m), X, y, splits, n_jobs=args.jobs, cache=cache)
        for m in args.models
    ]
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
        metadata={
            "cv": cv.summary(),
            "data": data_fingerprint(train),
            "first_month_q99": float(X["first_month"].quantile(0.99)),
        },
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
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="add 1 - alpha prediction intervals from the model's residuals",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="check the batch against the scoring contract and stop on errors",
    )


@command("predict", "score customers with a saved model", _predict_args)
def _predict(args: argparse.Namespace) -> int:
    import pandas as pd

    from loyalty_spend.data.io import load_clean
    from loyalty_spend.data.schema import ID, TARGET
    from loyalty_spend.models.persistence import load_bundle

    bundle = load_bundle(args.model)
    frame, _ = load_clean(args.input, impute=False, require_target=False)
    if args.alpha is not None or args.validate:
        from loyalty_spend.data.contracts import load_contracts
        from loyalty_spend.ops.batch import score_batch

        contract = load_contracts()["scoring"] if args.validate else None
        result = score_batch(bundle, frame, args.alpha or 0.1, contract)
        out = result.scores.rename(columns={"prediction": TARGET}).round(2)
        if args.alpha is None:
            out = out[[ID, TARGET]]
    else:
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
    p.add_argument(
        "--max-memory-mb",
        type=float,
        default=None,
        help="cap on the memory of one batch of draws, for large files",
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
            max_memory_mb=args.max_memory_mb,
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
            max_memory_mb=args.max_memory_mb,
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
    "design",
    "bayes",
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
