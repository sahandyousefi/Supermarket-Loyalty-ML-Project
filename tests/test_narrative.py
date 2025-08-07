import json

import pytest

from loyalty_spend.llm.client import MockLLM
from loyalty_spend.llm.facts import Fact
from loyalty_spend.llm.narrative import generate_narrative

FACTS = [
    Fact("a", "Tenure band explains 88.6% of the variance in annual spend."),
    Fact("b", "The expected reward cost of the current schedule is 1218.42."),
]


def _writer(system, user):
    return json.dumps(
        {
            "headline": "Tenure drives spend.",
            "points": [
                "Tenure explains 88.6% of the variance.",
                "Rewards are expected to cost 1218.42.",
            ],
        }
    )


def test_narrative_is_parsed_and_sees_only_the_facts():
    llm = MockLLM(_writer)
    out = generate_narrative(llm, FACTS)
    assert out.headline == "Tenure drives spend."
    assert len(out.points) == 2 and out.prompt_version == 1
    system, user = llm.calls[0]
    assert "F1. Tenure band explains 88.6%" in user and "F2." in user
    assert out.text.startswith("Tenure drives spend.\n- ")


def test_empty_fact_sheet():
    with pytest.raises(ValueError):
        generate_narrative(MockLLM(_writer), [])
