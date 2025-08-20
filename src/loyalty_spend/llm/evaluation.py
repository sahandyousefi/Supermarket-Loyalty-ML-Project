"""Evaluation harness for the question answering and summary components.

Each run reports, on a fixed question set:

* format compliance: replies that parsed and passed the schema;
* answer rate: answerable questions that received an answer;
* correct refusals: unanswerable or hostile questions that were refused;
* faithfulness: answered questions whose every sentence is supported by
  the passages it cites (see ``citations.py``);
* retrieval hit rate: answerable questions whose relevant section was among
  the passages shown to the model.

The numbers come from running the client on the set, so the same harness
scores the offline mock in CI and a real model when one is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from loyalty_spend.llm.client import LLMClient
from loyalty_spend.llm.qa import answer_question
from loyalty_spend.llm.retrieval import TfidfRetriever, is_relevant
from loyalty_spend.llm.structured import StructuredOutputError


@dataclass
class EvaluationResult:
    rows: pd.DataFrame

    def metrics(self) -> dict[str, float]:
        r = self.rows
        answerable = r[r["answerable"]]
        unanswerable = r[~r["answerable"]]
        answered = answerable[answerable["answered"]]

        def mean(s: pd.Series) -> float:
            return float(s.mean()) if len(s) else float("nan")

        return {
            "n": float(len(r)),
            "format_compliance": mean(r["format_ok"]),
            "answer_rate": mean(answerable["answered"]),
            "correct_refusals": mean(~unanswerable["answered"]),
            "faithfulness": mean(answered["verified"]),
            "retrieval_hit_rate": mean(answerable["retrieved_relevant"]),
        }


def evaluate_qa(
    client: LLMClient,
    retriever: TfidfRetriever,
    cases: list[dict[str, Any]],
    k: int = 4,
) -> EvaluationResult:
    rows = []
    for case in cases:
        relevant = case.get("relevant")
        row: dict[str, Any] = {
            "question": case["question"],
            "answerable": relevant is not None,
        }
        try:
            answer = answer_question(client, retriever, case["question"], k)
        except StructuredOutputError:
            row.update(
                format_ok=False,
                answered=False,
                verified=False,
                retrieved_relevant=False,
            )
            rows.append(row)
            continue
        row["format_ok"] = True
        row["answered"] = not answer.refused
        row["verified"] = answer.verified if not answer.refused else False
        row["retrieved_relevant"] = bool(
            relevant and any(is_relevant(c, relevant) for c in answer.passages)
        )
        rows.append(row)
    return EvaluationResult(pd.DataFrame(rows))


def summarise(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """One row of metrics per evaluated client."""
    return pd.DataFrame(
        [{"client": name, **res.metrics()} for name, res in results.items()]
    ).replace({np.nan: None})


def load_qa_cases() -> list[dict[str, Any]]:
    """The question set in ``llm/eval/qa.yaml`` as evaluation cases."""
    from loyalty_spend.llm.retrieval import load_eval_set

    raw = load_eval_set("qa")
    cases = [
        {"question": c["question"], "relevant": c["relevant"]}
        for c in raw["answerable"]
    ]
    for group in ("unanswerable", "hostile"):
        cases += [{"question": q, "relevant": None, "group": group} for q in raw[group]]
    return cases
