"""Token, latency and cost accounting for any client.

``TrackedClient`` wraps a client and records every call, so the evaluation
harness and the commands can report what a run cost. Prices are given per
thousand tokens because providers quote them that way; they default to
zero for local or mock models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from loyalty_spend.llm.client import LLMClient, LLMResponse


@dataclass
class TrackedClient:
    client: LLMClient
    prompt_price_per_1k: float = 0.0
    completion_price_per_1k: float = 0.0
    responses: list[LLMResponse] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.client.model

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        response = self.client.complete(
            system, user, temperature, max_tokens, json_mode
        )
        self.responses.append(response)
        return response

    def summary(self) -> dict[str, float]:
        prompt = sum(r.prompt_tokens for r in self.responses)
        completion = sum(r.completion_tokens for r in self.responses)
        latency = [r.latency_s for r in self.responses]
        cost = (
            prompt / 1000 * self.prompt_price_per_1k
            + completion / 1000 * self.completion_price_per_1k
        )
        return {
            "calls": float(len(self.responses)),
            "prompt_tokens": float(prompt),
            "completion_tokens": float(completion),
            "cost": float(cost),
            "latency_p50_s": float(np.median(latency)) if latency else 0.0,
            "latency_max_s": float(max(latency)) if latency else 0.0,
        }
