"""Versioned prompt templates stored as YAML files next to this module.

Prompts are part of the model: a change to their wording can change every
output, so each file carries a version that is recorded with the outputs
and checked by the evaluation harness.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

PROMPT_DIR = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str
    description: str = ""

    @property
    def fields(self) -> set[str]:
        return {f for _, f, _, _ in string.Formatter().parse(self.user) if f}

    def render(self, **values: object) -> tuple[str, str]:
        missing = self.fields - set(values)
        if missing:
            raise KeyError(f"prompt '{self.name}' needs {sorted(missing)}")
        return self.system.strip(), self.user.format(**values).strip()


@cache
def load_prompt(name: str) -> PromptTemplate:
    path = PROMPT_DIR / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"no prompt named '{name}'")
    raw = yaml.safe_load(path.read_text())
    return PromptTemplate(
        name=raw["name"],
        version=int(raw["version"]),
        system=raw["system"],
        user=raw["user"],
        description=raw.get("description", ""),
    )


def available_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPT_DIR.glob("*.yaml"))
