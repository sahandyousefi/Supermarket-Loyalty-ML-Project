import json
from pathlib import Path

from loyalty_spend.llm.facts import build_fact_sheet, render_facts

SUMMARY = {
    "drivers": {
        "eta_squared": 0.8858,
        "lmg": [{"feature": "loyalty_years", "relative_share": 0.88}],
    },
    "benchmark": [
        {"model": "ols", "rmse_mean": 0.30815, "p_value": float("nan")},
        {"model": "ridge", "rmse_mean": 0.30816, "p_value": 0.85},
        {"model": "mlp", "rmse_mean": 0.41, "p_value": 0.001},
    ],
    "promotion": {
        "estimates": [
            {
                "method": "AIPW",
                "estimand": "ate",
                "estimate": -0.015,
                "ci_lower": -0.056,
                "ci_upper": 0.026,
            }
        ]
    },
    "design": {
        "schedules": [{"schedule": "whole spend", "expected_cost": 1218.4228}],
        "tail_risk": {"var": 1223.47},
        "cost_to_spend": 0.0379,
    },
}


def test_fact_sheet_from_a_summary():
    facts = {f.key: f.statement for f in build_fact_sheet(SUMMARY)}
    assert (
        facts["tenure_eta2"]
        == "Tenure band explains 88.6% of the variance in annual spend."
    )
    assert "ols" in facts["best_model"] and "0.308" in facts["best_model"]
    assert facts["ties"].startswith("1 models")
    assert "-0.015" in facts["promotion_effect"]
    assert "1218.42" in facts["expected_cost"]
    assert render_facts(build_fact_sheet(SUMMARY)).startswith("F1. ")


def test_missing_sections_are_skipped():
    assert build_fact_sheet({}) == []
    assert [f.key for f in build_fact_sheet({"design": {"cost_to_spend": 0.05}})] == [
        "cost_ratio"
    ]


def test_committed_report_summary_gives_facts():
    summary = json.loads(Path("reports/summary.json").read_text())
    assert len(build_fact_sheet(summary)) >= 3
