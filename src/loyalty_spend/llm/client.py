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
    @property
    def model(self) -> str: ...

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


class HTTPChatClient:
    """Client for any server that implements the chat completions protocol.

    Configured by ``LLM_BASE_URL``, ``LLM_API_KEY`` and ``LLM_MODEL`` unless
    the values are passed in. Uses only the standard library. Rate limits
    (429) and server errors (5xx) are retried with exponential backoff; other
    errors are raised at once. The key is read at call time and never
    stored in logs or responses.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 1.0,
    ) -> None:
        import os

        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        if not self.base_url or not self.model:
            raise ValueError("set LLM_BASE_URL and LLM_MODEL or pass them in")

    def _key(self) -> str:
        import os

        return self._api_key or os.environ.get("LLM_API_KEY", "")

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        import json
        import urllib.error
        import urllib.request

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}",
            },
            method="POST",
        )
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as err:
                retryable = err.code == 429 or err.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * 2**attempt)
        text = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage", {})
        return LLMResponse(
            text=text,
            model=body.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", approx_tokens(system + user))),
            completion_tokens=int(usage.get("completion_tokens", approx_tokens(text))),
            latency_s=time.perf_counter() - start,
        )


def client_from_env() -> HTTPChatClient | None:
    """A configured HTTP client, or ``None`` when ``LLM_BASE_URL`` is not set."""
    import os

    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_MODEL"):
        return None
    return HTTPChatClient()
