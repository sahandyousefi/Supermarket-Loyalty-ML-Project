"""Name-based registry of model factories.

Model modules register their factories with the ``register`` decorator, so a
new family becomes available to the CLI, the comparison and the tuner without
editing a central table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_MODULES = (
    "loyalty_spend.models.linear",
    "loyalty_spend.models.trees",
    "loyalty_spend.models.deep.estimator",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[..., Any]
    requires: str | None
    family: str


_REGISTRY: dict[str, ModelSpec] = {}


def register(
    name: str, family: str, requires: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``factory`` under ``name``.

    ``requires`` names an optional package; the model is only listed as
    available when that package can be imported.
    """

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' is already registered")
        _REGISTRY[name] = ModelSpec(name, factory, requires, family)
        return factory

    return decorator
