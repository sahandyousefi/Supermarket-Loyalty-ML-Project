"""Verification that cited passages support the sentences that cite them.

A citation can be present and still wrong. Each sentence of an answer is
checked against the passages it cites: every number in the sentence must
occur in one of them, and enough of its content words must occur there
too. The check is lexical and deliberately strict, so it flags paraphrases
as well as fabrications; flagged sentences are for a person to review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loyalty_spend.llm.corpus import Chunk
from loyalty_spend.llm.grounding import extract_numbers

CITE = re.compile(r"\[(\d+)\]")
WORD = re.compile(r"[a-z][a-z\-]{2,}")
STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "are",
        "with",
        "that",
        "this",
        "from",
        "which",
        "have",
        "has",
        "was",
        "were",
        "not",
        "but",
        "its",
        "their",
        "than",
        "into",
        "over",
        "under",
        "each",
        "when",
        "where",
        "what",
        "there",
        "these",
        "those",
        "been",
        "being",
        "also",
        "only",
        "more",
        "most",
        "less",
        "such",
        "they",
        "them",
        "then",
        "than",
        "can",
        "may",
        "will",
        "would",
        "should",
    ]
)


@dataclass(frozen=True)
class SentenceCheck:
    sentence: str
    cited: tuple[int, ...]
    overlap: float
    missing_numbers: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return bool(self.cited) and not self.missing_numbers and self.overlap >= 0.5


def _content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [p for p in parts if p]


def verify_citations(answer: str, passages: list[Chunk]) -> list[SentenceCheck]:
    checks = []
    for sentence in split_sentences(answer):
        cited = tuple(
            int(n) for n in CITE.findall(sentence) if 1 <= int(n) <= len(passages)
        )
        claim = CITE.sub("", sentence)
        source = " ".join(passages[i - 1].text for i in cited)
        words = _content_words(claim)
        overlap = len(words & _content_words(source)) / len(words) if words else 1.0
        source_numbers = set(extract_numbers(source))
        missing = tuple(n for n in extract_numbers(claim) if n not in source_numbers)
        checks.append(SentenceCheck(sentence, cited, overlap, missing))
    return checks
