"""Lexical retrieval over the documentation chunks.

Lexical retrieval is enough for a corpus of a few hundred chunks written in
consistent terminology, and it is transparent: a passage is retrieved
because it shares weighted terms with the question. Okapi BM25 (Robertson
and Zaragoza, 2009) is the default. Unlike cosine-normalised TF-IDF it
saturates repeated terms and normalises for chunk length only in part, so
long sections that mention a term in passing do not crowd out the section
about it. TF-IDF is kept for comparison. Retrieval quality is measured on a
fixed set of questions with known relevant sections by recall at k (is a
relevant chunk among the first k) and mean reciprocal rank (how high the
first relevant chunk is ranked).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from loyalty_spend.llm.corpus import Chunk

EVAL_DIR = Path(__file__).with_name("eval")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    chunks: list[Chunk]

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def _document(chunk: Chunk) -> str:
    # The heading is repeated so that it weighs more than a body sentence.
    return f"{chunk.heading}\n{chunk.heading}\n{chunk.text}"


class TfidfRetriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), stop_words="english", min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(map(_document, self.chunks))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


class Bm25Retriever:
    """Okapi BM25 with the usual defaults :math:`k_1 = 1.2`, :math:`b = 0.75`."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75):
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self.chunks = list(chunks)
        self.vectorizer = CountVectorizer(stop_words="english")
        tf = self.vectorizer.fit_transform(map(_document, self.chunks)).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        length = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * length / length.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        weights = tf.astype(float)
        weights.data = weights.data * (k1 + 1) / (weights.data + norm[rows])
        self.matrix = weights

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.vectorizer.transform([query])
        q.data = np.ones_like(q.data, dtype=float)
        scores = (self.matrix @ q.multiply(self.idf).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def is_relevant(chunk: Chunk, target: dict[str, str]) -> bool:
    """A chunk is relevant if it comes from the target source and heading."""
    return (
        chunk.source == target["source"]
        and target["heading"].lower() in chunk.heading.lower()
    )


def retrieval_metrics(
    retriever: Retriever, items: list[dict], k: int = 5
) -> dict[str, float]:
    recalls, reciprocal = [], []
    for item in items:
        hits = retriever.search(item["question"], k)
        ranks = [
            i
            for i, h in enumerate(hits, start=1)
            if is_relevant(h.chunk, item["relevant"])
        ]
        recalls.append(1.0 if ranks else 0.0)
        reciprocal.append(1.0 / ranks[0] if ranks else 0.0)
    return {
        f"recall_at_{k}": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal)),
        "n": len(items),
    }


def load_eval_set(name: str = "retrieval") -> Any:
    """Parsed YAML of an evaluation file in ``llm/eval``."""
    return yaml.safe_load((EVAL_DIR / f"{name}.yaml").read_text())
