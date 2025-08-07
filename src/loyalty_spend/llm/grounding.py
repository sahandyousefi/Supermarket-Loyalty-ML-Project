"""Check that every number in generated text comes from the fact sheet.

Language models invent and recompute numbers. Since every fact already
states its numbers in final form, a faithful summary can only repeat them;
any other number in the output is unsupported. Percentages are compared
with percentages and plain numbers with plain numbers, after removing
thousands separators, so "1,218.42" matches "1218.42" but "88.6%" does not
match "88.6".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loyalty_spend.llm.facts import Fact

NUMBER = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(text: str) -> list[str]:
    """Numbers in ``text`` in a normal form: no separators, ``-`` sign, ``%`` kept."""
    out = []
    for raw in NUMBER.findall(text):
        value = raw.replace(",", "").replace("−", "-").rstrip(".")
        out.append(value)
    return out


def _key(value: str) -> tuple[float, bool]:
    percent = value.endswith("%")
    return float(value.rstrip("%")), percent


@dataclass
class GroundingResult:
    unsupported: list[str]
    checked: int

    @property
    def grounded(self) -> bool:
        return not self.unsupported


def check_numbers(
    text: str, facts: list[Fact], allowed: set[str] | None = None
) -> GroundingResult:
    """Numbers in ``text`` that appear in no fact.

    ``allowed`` lists numbers that may appear without a fact, such as the
    levels of a categorical attribute ("10+" years) quoted by name.
    """
    known = {_key(v) for f in facts for v in extract_numbers(f.statement)}
    known |= {_key(v) for v in allowed or set()}
    found = extract_numbers(text)
    unsupported = [v for v in found if _key(v) not in known]
    return GroundingResult(unsupported, len(found))
