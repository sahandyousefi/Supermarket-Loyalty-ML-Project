import pytest

from loyalty_spend.llm.corpus import Chunk, load_corpus
from loyalty_spend.llm.retrieval import TfidfRetriever, load_eval_set, retrieval_metrics


@pytest.fixture(scope="module")
def retriever():
    return TfidfRetriever(load_corpus())


def test_search_ranks_the_matching_chunk_first():
    chunks = [
        Chunk("a", "x.md", "Cats", "Cats purr and sleep."),
        Chunk("b", "x.md", "Bonds", "Bond prices fall when yields rise."),
    ]
    hits = TfidfRetriever(chunks).search("why do bond prices fall", k=2)
    assert hits[0].chunk.id == "b"
    assert all(h.score > 0 for h in hits)
    with pytest.raises(ValueError):
        TfidfRetriever([])


def test_retrieval_quality_on_the_evaluation_set(retriever):
    items = load_eval_set()
    metrics = retrieval_metrics(retriever, items, k=5)
    assert metrics["n"] == len(items) >= 10
    assert metrics["recall_at_5"] >= 0.8
    assert metrics["mrr"] >= 0.5
