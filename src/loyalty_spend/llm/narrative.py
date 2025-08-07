"""Executive summary written by a language model from the fact sheet."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loyalty_spend.llm.client import LLMClient
from loyalty_spend.llm.facts import Fact, render_facts
from loyalty_spend.llm.prompts import load_prompt


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


def generate_narrative(client: LLMClient, facts: list[Fact]) -> Narrative:
    """Ask ``client`` for a summary of ``facts`` and parse its JSON reply."""
    if not facts:
        raise ValueError("cannot summarise an empty fact sheet")
    prompt = load_prompt("narrative")
    system, user = prompt.render(facts=render_facts(facts))
    response = client.complete(system, user, json_mode=True)
    data = json.loads(response.text)
    return Narrative(
        headline=str(data["headline"]),
        points=[str(p) for p in data["points"]],
        model=response.model,
        prompt_version=prompt.version,
        raw=response.text,
    )
