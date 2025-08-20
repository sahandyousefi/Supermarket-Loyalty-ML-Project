"""Executive summary written by a language model from the fact sheet."""

from __future__ import annotations

from dataclasses import dataclass, field

from loyalty_spend.llm.client import LLMClient
from loyalty_spend.llm.facts import Fact, render_facts
from loyalty_spend.llm.grounding import check_numbers
from loyalty_spend.llm.prompts import load_prompt
from loyalty_spend.llm.structured import complete_json


@dataclass
class Narrative:
    headline: str
    points: list[str]
    model: str
    prompt_version: int
    raw: str = field(repr=False, default="")

    @property
    def text(self) -> str:
        return "\n".join([self.headline, *(f"- {p}" for p in self.points)])


def generate_narrative(
    client: LLMClient, facts: list[Fact], max_attempts: int = 2
) -> Narrative:
    """Ask ``client`` for a summary of ``facts``.

    The reply must be valid JSON with a headline and a list of points, and
    every number in it must come from the fact sheet; otherwise the model is
    asked again with the problem stated, up to ``max_attempts`` times.
    """
    if not facts:
        raise ValueError("cannot summarise an empty fact sheet")
    prompt = load_prompt("narrative")
    system, user = prompt.render(facts=render_facts(facts))

    def numbers_from_facts(data: dict) -> str | None:
        text = " ".join([str(data["headline"]), *map(str, data["points"])])
        result = check_numbers(text, facts)
        if result.grounded:
            return None
        return f"these numbers are not in the fact sheet: {result.unsupported}"

    reply = complete_json(
        client,
        system,
        user,
        {"headline": str, "points": list},
        validator=numbers_from_facts,
        max_attempts=max_attempts,
    )
    return Narrative(
        headline=str(reply.data["headline"]),
        points=[str(p) for p in reply.data["points"]],
        model=reply.response.model,
        prompt_version=prompt.version,
        raw=reply.response.text,
    )


def template_narrative(facts: list[Fact]) -> Narrative:
    """Summary without a language model: the facts themselves, in order."""
    if not facts:
        raise ValueError("cannot summarise an empty fact sheet")
    statements = [f.statement for f in facts]
    return Narrative(statements[0], statements[1:6], "template", 0)
