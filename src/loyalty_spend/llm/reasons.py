"""Plain-language reasons for a single spend forecast.

The numbers come from the exact additive contributions of the linear model:
the forecast equals the average forecast over a reference population plus
one contribution per attribute. The language model only turns the largest
contributions into sentences, and the grounding check rejects any number
that is not one of them. A template version gives the same content without
a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from loyalty_spend.explain.attribution import _group
from loyalty_spend.llm.client import LLMClient
from loyalty_spend.llm.facts import Fact
from loyalty_spend.llm.grounding import check_numbers
from loyalty_spend.llm.prompts import load_prompt
from loyalty_spend.llm.structured import complete_json

LABELS = {
    "loyalty_years": "tenure band",
    "first_month": "first-month spend",
    "items_in_first_month": "first-month basket size",
    "region": "region",
    "joining_month": "joining month",
    "promotion": "sign-up promotion",
}


@dataclass
class ForecastReasons:
    baseline: float
    prediction: float
    contributions: pd.Series

    def top(self, n: int = 3) -> pd.Series:
        order = self.contributions.abs().sort_values(ascending=False).index[:n]
        return self.contributions[order]

    def facts(self, n: int = 3) -> list[Fact]:
        rows = [
            Fact("baseline", f"Average forecast: {self.baseline:.2f}"),
            Fact("prediction", f"This member's forecast: {self.prediction:.2f}"),
        ]
        for name, value in self.top(n).items():
            rows.append(Fact(name, f"{LABELS.get(name, name)}: {value:+.2f}"))
        return rows


def forecast_reasons(
    pipeline: Pipeline, reference: pd.DataFrame, row: pd.DataFrame
) -> ForecastReasons:
    """Contributions of each attribute to the forecast for one ``row``."""
    prep, model = pipeline[:-1], pipeline[-1]
    z_ref = np.asarray(prep.transform(reference), float)
    z = np.asarray(prep.transform(row), float)
    names = list(prep.get_feature_names_out())
    values = (z - z_ref.mean(axis=0)) * np.ravel(model.coef_)
    contributions = _group(values, names).iloc[0]
    baseline = float(pipeline.predict(reference).mean())
    return ForecastReasons(baseline, float(pipeline.predict(row)[0]), contributions)


def template_explanation(reasons: ForecastReasons, n: int = 3) -> str:
    parts = [
        f"{LABELS.get(k, k)} {'adds' if v >= 0 else 'subtracts'} {abs(v):.2f}"
        for k, v in reasons.top(n).items()
    ]
    return (
        f"The forecast of {reasons.prediction:.2f} compares with an average of "
        f"{reasons.baseline:.2f}: " + "; ".join(parts) + "."
    )


def explain_forecast(client: LLMClient, reasons: ForecastReasons, n: int = 3) -> str:
    facts = reasons.facts(n)
    lines = [f.statement for f in facts[2:]]
    system, user = load_prompt("reasons").render(
        baseline=f"{reasons.baseline:.2f}",
        prediction=f"{reasons.prediction:.2f}",
        contributions="\n".join(lines),
    )
    allowed = {f"{abs(v):.2f}" for v in reasons.top(n)}

    def grounded(data: dict) -> str | None:
        result = check_numbers(data["explanation"], facts, allowed)
        return None if result.grounded else f"unsupported numbers {result.unsupported}"

    reply = complete_json(client, system, user, {"explanation": str}, grounded)
    return str(reply.data["explanation"])
