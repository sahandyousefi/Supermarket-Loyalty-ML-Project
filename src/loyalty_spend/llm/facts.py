"""Fact sheets: the only material a language model may summarise.

The fact sheet is built in code from the report's ``summary.json``. Each fact
is one sentence with its numbers already formatted, so the model never has
to compute, round or compare anything; the grounding check then compares
every number in the generated text with the numbers in this sheet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Fact:
    key: str
    statement: str


def _drivers(d: dict[str, Any]) -> list[Fact]:
    facts = []
    if "eta_squared" in d:
        facts.append(
            Fact(
                "tenure_eta2",
                f"Tenure band explains {d['eta_squared']:.1%} of the variance "
                "in annual spend.",
            )
        )
    lmg = d.get("lmg") or []
    if lmg:
        top = max(lmg, key=lambda r: r["relative_share"])
        facts.append(
            Fact(
                "top_driver",
                "The largest share of explained variance, "
                f"{top['relative_share']:.1%}, belongs to {top['feature']}.",
            )
        )
    return facts


def _benchmark(rows: list[dict[str, Any]]) -> list[Fact]:
    if not rows:
        return []
    best = min(rows, key=lambda r: r["rmse_mean"])
    facts = [
        Fact(
            "best_model",
            f"The most accurate model is {best['model']} with a cross-validated RMSE "
            f"of {best['rmse_mean']:.3f}.",
        ),
        Fact("n_models", f"{len(rows)} models were compared on the same folds."),
    ]
    tied = [
        r for r in rows if r.get("p_value", 0.0) >= 0.05 and r["model"] != best["model"]
    ]
    if tied:
        facts.append(
            Fact(
                "ties",
                f"{len(tied)} models are not significantly different from the OLS "
                "reference at the 5% level.",
            )
        )
    return facts


def _promotion(d: dict[str, Any]) -> list[Fact]:
    for row in d.get("estimates", []):
        if row["method"] == "AIPW" and row["estimand"] == "ate":
            return [
                Fact(
                    "promotion_effect",
                    "The estimated effect of the sign-up promotion on annual spend "
                    f"is {row['estimate']:.3f}, with a 95% interval from "
                    f"{row['ci_lower']:.3f} to {row['ci_upper']:.3f}.",
                )
            ]
    return []


def _design(d: dict[str, Any]) -> list[Fact]:
    facts = []
    risk = d.get("tail_risk") or {}
    rows = {r["schedule"]: r for r in d.get("schedules", [])}
    if "whole spend" in rows:
        facts.append(
            Fact(
                "expected_cost",
                f"The expected reward cost of the current schedule is "
                f"{rows['whole spend']['expected_cost']:.2f}.",
            )
        )
    if risk:
        facts.append(
            Fact(
                "cost_var",
                f"With 95% probability the reward cost stays below {risk['var']:.2f}.",
            )
        )
    if "cost_to_spend" in d:
        facts.append(
            Fact(
                "cost_ratio",
                f"Rewards cost {d['cost_to_spend']:.2%} of expected spend.",
            )
        )
    return facts


SECTIONS: dict[str, Callable[[Any], list[Fact]]] = {
    "drivers": _drivers,
    "benchmark": _benchmark,
    "promotion": _promotion,
    "design": _design,
}


def build_fact_sheet(summary: dict[str, Any]) -> list[Fact]:
    """Facts for every report section present in ``summary``."""
    facts: list[Fact] = []
    for name, builder in SECTIONS.items():
        if name in summary:
            facts.extend(builder(summary[name]))
    return facts


def render_facts(facts: list[Fact]) -> str:
    return "\n".join(f"F{i}. {f.statement}" for i, f in enumerate(facts, start=1))
