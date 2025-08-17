import json

import pytest

from loyalty_spend.llm.client import MockLLM
from loyalty_spend.llm.corpus import Chunk
from loyalty_spend.llm.qa import REFUSAL, answer_question
from loyalty_spend.llm.retrieval import TfidfRetriever

CHUNKS = [
    Chunk(
        "a",
        "docs/x.md",
        "Intervals",
        "Split conformal intervals cover 90% of new outcomes.",
    ),
    Chunk("b", "docs/x.md", "Costs", "The reward cost is simulated by Monte Carlo."),
]


@pytest.fixture
def retriever():
    return TfidfRetriever(CHUNKS)


def test_answer_with_citations(retriever):
    llm = MockLLM(
        lambda s, u: json.dumps({"answer": "They cover 90% [1].", "citations": [1]})
    )
    out = answer_question(
        llm, retriever, "How often do conformal intervals cover?", k=2
    )
    assert out.sources[0].id == "a" and not out.refused
    assert "[1] (docs/x.md, Intervals)" in llm.calls[0][1]


def test_invalid_citation_is_retried(retriever):
    replies = iter(
        [
            json.dumps({"answer": "Yes [7].", "citations": [7]}),
            json.dumps({"answer": REFUSAL, "citations": []}),
        ]
    )
    llm = MockLLM(lambda s, u: next(replies))
    out = answer_question(llm, retriever, "conformal coverage", k=2)
    assert out.refused and out.sources == []
    assert "do not refer" in llm.calls[1][1]


def test_no_matching_passage_refuses_without_a_call(retriever):
    llm = MockLLM()
    assert answer_question(llm, retriever, "zebra migration").refused
    assert llm.calls == []


def test_answer_verification(retriever):
    llm = MockLLM(
        lambda s, u: json.dumps(
            {
                "answer": "Split conformal intervals cover 90% of new outcomes [1].",
                "citations": [1],
            }
        )
    )
    out = answer_question(llm, retriever, "conformal coverage", k=2)
    assert out.verified
