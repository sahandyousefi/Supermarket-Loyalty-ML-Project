import json

import pytest

from loyalty_spend.llm.client import MockLLM
from loyalty_spend.llm.facts import Fact
from loyalty_spend.llm.narrative import generate_narrative
from loyalty_spend.llm.structured import (
    StructuredOutputError,
    complete_json,
    schema_problem,
)


def _sequence(*replies):
    it = iter(replies)
    return MockLLM(lambda system, user: next(it))


def test_schema_problems():
    assert schema_problem([], {"a": str}) == "the reply must be a JSON object"
    assert "missing" in schema_problem({}, {"a": str})
    assert "type list" in schema_problem({"a": "x"}, {"a": list})
    assert schema_problem({"a": "x"}, {"a": str}) is None


def test_retry_after_invalid_json_then_success():
    llm = _sequence("not json", json.dumps({"answer": "ok"}))
    reply = complete_json(llm, "s", "u", {"answer": str})
    assert reply.data == {"answer": "ok"} and reply.attempts == 2
    assert "not valid JSON" in llm.calls[1][1]


def test_gives_up_after_the_bound():
    llm = _sequence("{}", "{}", "{}")
    with pytest.raises(StructuredOutputError, match="after 2 attempts"):
        complete_json(llm, "s", "u", {"answer": str})
    assert len(llm.calls) == 2


def test_narrative_rejects_invented_numbers_then_accepts_a_faithful_reply():
    facts = [Fact("a", "Rewards cost 3.79% of expected spend.")]
    bad = json.dumps({"headline": "Rewards cost 4% of spend.", "points": []})
    good = json.dumps(
        {"headline": "Rewards cost 3.79% of expected spend.", "points": []}
    )
    llm = _sequence(bad, good)
    out = generate_narrative(llm, facts)
    assert out.headline.startswith("Rewards cost 3.79%")
    assert "4%" in llm.calls[1][1]
