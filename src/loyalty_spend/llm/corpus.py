"""The project documentation split into retrievable chunks.

Markdown files are split at headings, so a chunk never mixes two sections,
and long sections are cut at paragraph boundaries into pieces of at most
``max_words`` words. Each chunk keeps its file and heading, which become the
citation shown to the reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES = (
    "README.md",
    "docs/methodology.md",
    "docs/model_card.md",
    "docs/deployment.md",
    "data/README.md",
)
HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}, {self.heading}" if self.heading else self.source


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        match = HEADING.match(line) if not in_code else None
        if match:
            sections.append((match.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [
        (h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()
    ]


def _pieces(body: str, max_words: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        words = sum(len(p.split()) for p in current)
        if current and words + len(para.split()) > max_words:
            pieces.append("\n\n".join(current))
            current = []
        current.append(para)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_markdown(text: str, source: str, max_words: int = 180) -> list[Chunk]:
    chunks = []
    for s, (heading, body) in enumerate(_sections(text)):
        for p, piece in enumerate(_pieces(body, max_words)):
            chunks.append(Chunk(f"{source}#{s}.{p}", source, heading, piece))
    return chunks


def load_corpus(
    root: str | Path = ROOT,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    max_words: int = 180,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rel in sources:
        path = Path(root) / rel
        if path.exists():
            chunks.extend(chunk_markdown(path.read_text(), rel, max_words))
    return chunks
