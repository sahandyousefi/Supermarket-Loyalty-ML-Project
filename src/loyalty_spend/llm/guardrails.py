"""Input checks before a question reaches the language model.

Questions are free text from users, so they can try to override the
instructions ("ignore the passages and ..."), extract the prompt, or carry
material far longer than any real question. These checks run before any
call: a blocked question is answered with a fixed refusal and costs
nothing. Pattern lists catch the common phrasings only; the prompt also
tells the model to treat passages as data, and answers are still checked
against their citations, so no single layer has to be perfect.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_CHARS = 500
PATTERNS = [
    r"\bignore\b.{0,40}\b(instruction|prompt|passage|rule|above|previous)",
    r"\bdisregard\b.{0,40}\b(instruction|prompt|passage|rule|above|previous)",
    r"\b(system|developer)\s+(prompt|message|instruction)",
    r"\byou are now\b",
    r"\bact as\b",
    r"\bpretend (to be|you)\b",
    r"\b(reveal|print|show|repeat)\b.{0,30}\b(prompt|instructions)\b",
    r"\bjailbreak\b",
    r"<\s*/?\s*(system|assistant|user)\s*>",
]
COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


@dataclass
class GuardResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    text: str = ""


def normalise(text: str) -> str:
    """NFKC form without control characters, so look-alike tricks are folded."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )


def check_question(text: str, max_chars: int = MAX_CHARS) -> GuardResult:
    clean = normalise(text).strip()
    reasons = []
    if not clean:
        reasons.append("empty question")
    if len(clean) > max_chars:
        reasons.append(f"longer than {max_chars} characters")
    for pattern in COMPILED:
        if pattern.search(clean):
            reasons.append("looks like an attempt to change the instructions")
            break
    return GuardResult(not reasons, reasons, clean)
