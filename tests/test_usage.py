import pytest

from loyalty_spend.llm.client import MockLLM
from loyalty_spend.llm.usage import TrackedClient


def test_tracked_client_counts_tokens_and_cost():
    tracked = TrackedClient(MockLLM(lambda s, u: "x" * 400), 0.5, 1.5)
    tracked.complete("s" * 40, "u" * 360)
    tracked.complete("s" * 40, "u" * 360)
    summary = tracked.summary()
    assert summary["calls"] == 2
    assert summary["prompt_tokens"] == 2 * (10 + 90)
    assert summary["completion_tokens"] == 200
    assert summary["cost"] == pytest.approx(0.2 * 0.5 + 0.2 * 1.5)
    assert tracked.model == "mock"


def test_empty_summary():
    assert TrackedClient(MockLLM()).summary()["calls"] == 0
