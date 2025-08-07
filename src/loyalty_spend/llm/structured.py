"""JSON replies with schema checks and a bounded number of retries.

A reply is accepted only if it parses as a JSON object, has every required
key with the right type and passes an optional content validator (for
example the numeric grounding check). Otherwise the model is asked once
more with the specific problem appended to the request. After
``max_attempts`` failures the error is raised instead of returning
unchecked text.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loyalty_spend.llm.client import LLMClient, LLMResponse

Validator = Callable[[dict[str, Any]], str | None]


class StructuredOutputError(ValueError):
    pass


@dataclass
class StructuredReply:
    data: dict[str, Any]
    response: LLMResponse
    attempts: int


def schema_problem(data: Any, required: dict[str, type]) -> str | None:
    if not isinstance(data, dict):
        return "the reply must be a JSON object"
    for key, kind in required.items():
        if key not in data:
            return f"the key '{key}' is missing"
        if not isinstance(data[key], kind):
            return f"'{key}' must be of type {kind.__name__}"
    return None


def complete_json(
    client: LLMClient,
    system: str,
    user: str,
    required: dict[str, type],
    validator: Validator | None = None,
    max_attempts: int = 2,
) -> StructuredReply:
    message = user
    problem = "no attempt made"
    for attempt in range(1, max_attempts + 1):
        response = client.complete(system, message, json_mode=True)
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            problem = "the reply was not valid JSON"
        else:
            problem = (
                schema_problem(data, required)
                or (validator(data) if validator else None)
                or ""
            )
            if not problem:
                return StructuredReply(data, response, attempt)
        message = (
            f"{user}\n\nYour previous reply was rejected because {problem}. "
            "Reply again with valid JSON that fixes this."
        )
    raise StructuredOutputError(
        f"no valid reply after {max_attempts} attempts: {problem}"
    )
