"""Language model clients behind one small interface.

Every component that writes text calls ``complete(system, user)`` on a
client. Tests and CI use :class:`MockLLM`, which is deterministic and needs
no network, so the grounding checks and the evaluation harness are tested
on fixed outputs. A real provider is plugged in through the same interface.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse: ...


def approx_tokens(text: str) -> int:
    """Rough token count, about four characters per token for English text."""
    return max(1, len(text) // 4)


Responder = Callable[[str, str], str]


def _echo(system: str, user: str) -> str:
    digest = hashlib.sha256((system + "\n" + user).encode()).hexdigest()[:8]
    return f"mock response {digest}"


@dataclass
class MockLLM:
    """Deterministic offline client.

    ``responder`` maps the system and user prompt to the reply; the default
    returns a short string tagged with a hash of the prompt. Every call is
    kept in ``calls`` so tests can inspect what was sent.
    """

    responder: Responder = _echo
    model: str = "mock"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        start = time.perf_counter()
        self.calls.append((system, user))
        text = self.responder(system, user)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=approx_tokens(system) + approx_tokens(user),
            completion_tokens=approx_tokens(text),
            latency_s=time.perf_counter() - start,
        )
