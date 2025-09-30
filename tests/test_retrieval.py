import pytest

from loyalty_spend.llm.corpus import Chunk, load_corpus
from loyalty_spend.llm.retrieval import (
    Bm25Retriever,
    TfidfRetriever,
    load_eval_set,
    retrieval_metrics,
)


@pytest.fixture(scope="module")
def retriever():
    return Bm25Retriever(load_corpus())


@pytest.mark.parametrize("cls", [TfidfRetriever, Bm25Retriever])
def test_search_ranks_the_matching_chunk_first(cls):
    chunks = [
        Chunk("a", "x.md", "Cats", "Cats purr and sleep."),
        Chunk("b", "x.md", "Bonds", "Bond prices fall when yields rise."),
    ]
    hits = cls(chunks).search("why do bond prices fall", k=2)
    assert hits[0].chunk.id == "b"
    assert all(h.score > 0 for h in hits)
    with pytest.raises(ValueError):
        cls([])


def test_retrieval_quality_on_the_evaluation_set(retriever):
    items = load_eval_set()
    metrics = retrieval_metrics(retriever, items, k=5)
    assert metrics["n"] == len(items) >= 10
    assert metrics["recall_at_5"] >= 0.8
    assert metrics["mrr"] >= 0.5


def test_bm25_is_at_least_as_good_as_tfidf(retriever):
    items = load_eval_set()
    bm25 = retrieval_metrics(retriever, items, k=5)
    tfidf = retrieval_metrics(TfidfRetriever(load_corpus()), items, k=5)
    assert bm25["recall_at_5"] >= tfidf["recall_at_5"]
    assert bm25["mrr"] >= tfidf["mrr"]


def test_bm25_saturates_repeated_terms():
    chunks = [
        Chunk("a", "x.md", "Noise", "tenure " * 50 + "unrelated words " * 50),
        Chunk("b", "x.md", "Tenure", "tenure drives spend"),
    ]
    hits = Bm25Retriever(chunks).search("tenure spend", k=2)
    assert hits[0].chunk.id == "b"
