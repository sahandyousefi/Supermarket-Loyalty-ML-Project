"""TF-IDF retrieval over the documentation chunks.

Lexical retrieval is enough for a corpus of a few hundred chunks written in
consistent terminology, and it is transparent: a passage is retrieved
because it shares weighted terms with the question. Retrieval quality is
measured on a fixed set of questions with known relevant sections by
recall at k (is a relevant chunk among the first k) and mean reciprocal
rank (how high the first relevant chunk is ranked).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from loyalty_spend.llm.corpus import Chunk

EVAL_DIR = Path(__file__).with_name("eval")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class TfidfRetriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), stop_words="english", min_df=1
        )
        texts = [f"{c.heading}\n{c.heading}\n{c.text}" for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def is_relevant(chunk: Chunk, target: dict[str, str]) -> bool:
    """A chunk is relevant if it comes from the target source and heading."""
    return (
        chunk.source == target["source"]
        and target["heading"].lower() in chunk.heading.lower()
    )


def retrieval_metrics(
    retriever: TfidfRetriever, items: list[dict], k: int = 5
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


def load_eval_set(name: str = "retrieval") -> list[dict]:
    return yaml.safe_load((EVAL_DIR / f"{name}.yaml").read_text())
