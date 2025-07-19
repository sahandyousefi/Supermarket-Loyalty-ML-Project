"""A file-based model registry.

Each registered model gets a numbered version folder with the bundle and a
JSON record of its name, metrics, training data fingerprint and stage. The
registry is a plain folder, so it can live next to the project, in a
mounted volume or in object storage synced to disk, without a server.

::

    registry/
      v0001/model.joblib
      v0001/record.json
      v0002/...
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loyalty_spend.models.persistence import ModelBundle, load_bundle, save_bundle

STAGES = ("none", "staging", "production", "archived")


@dataclass
class ModelRecord:
    version: int
    name: str
    stage: str = "none"
    metrics: dict[str, float] = field(default_factory=dict)
    data_fingerprint: str = ""
    registered_at: str = ""
    notes: str = ""

    @property
    def tag(self) -> str:
        return f"v{self.version:04d}"


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, version: int) -> Path:
        return self.root / f"v{version:04d}"

    def _write(self, record: ModelRecord) -> None:
        path = self._dir(record.version) / "record.json"
        path.write_text(json.dumps(asdict(record), indent=2))

    def records(self) -> list[ModelRecord]:
        out = []
        for path in sorted(self.root.glob("v*/record.json")):
            out.append(ModelRecord(**json.loads(path.read_text())))
        return out

    def get(self, version: int) -> ModelRecord:
        path = self._dir(version) / "record.json"
        if not path.exists():
            raise KeyError(f"no model version {version}")
        return ModelRecord(**json.loads(path.read_text()))

    def register(
        self,
        bundle: ModelBundle,
        metrics: dict[str, float] | None = None,
        notes: str = "",
    ) -> ModelRecord:
        """Store ``bundle`` as the next version."""
        versions = [r.version for r in self.records()]
        version = max(versions, default=0) + 1
        save_bundle(bundle, self._dir(version) / "model.joblib")
        record = ModelRecord(
            version=version,
            name=bundle.name,
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            data_fingerprint=str(bundle.metadata.get("data_fingerprint", "")),
            registered_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )
        self._write(record)
        return record

    def load(self, version: int) -> ModelBundle:
        self.get(version)
        return load_bundle(self._dir(version) / "model.joblib")

    def promote(self, version: int, stage: str = "production") -> ModelRecord:
        """Move ``version`` to ``stage``.

        Only one version can be in production; the version it replaces is
        archived, with a note of when, so that it can be restored.
        """
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        record = self.get(version)
        if stage == "production":
            for other in self.records():
                if other.stage == "production" and other.version != version:
                    other.stage = "archived"
                    other.notes = (other.notes + f" replaced by v{version:04d}").strip()
                    self._write(other)
        record.stage = stage
        self._write(record)
        return record

    def current(self, stage: str = "production") -> ModelRecord | None:
        matches = [r for r in self.records() if r.stage == stage]
        return matches[-1] if matches else None

    def rollback(self) -> ModelRecord:
        """Restore the most recently archived version to production."""
        current = self.current("production")
        archived = [r for r in self.records() if r.stage == "archived"]
        if current is not None:
            archived = [r for r in archived if r.version != current.version]
        if not archived:
            raise RuntimeError("no archived version to roll back to")
        target = max(archived, key=lambda r: r.version)
        self.promote(target.version, "production")
        if current is not None:
            current = self.get(current.version)
            current.stage = "archived"
            current.notes = (current.notes + " rolled back").strip()
            self._write(current)
        return self.get(target.version)

    def summary(self) -> list[dict[str, Any]]:
        return [{**asdict(r), "tag": r.tag} for r in self.records()]


def promote_if_better(
    registry: ModelRegistry,
    challenger_version: int,
    X: Any,
    y: Any,
    margin: float = 0.0,
    seed: int = 0,
) -> tuple[bool, dict[str, float]]:
    """Promote ``challenger_version`` to production only if it beats the champion.

    Without a production model the challenger is promoted directly. The
    comparison and the decision are written into the challenger's record.
    """
    from loyalty_spend.ops.champion import challenge

    champion = registry.current("production")
    record = registry.get(challenger_version)
    if champion is None:
        registry.promote(challenger_version)
        return True, {}
    result = challenge(
        registry.load(champion.version),
        registry.load(challenger_version),
        X,
        y,
        margin,
        seed,
    )
    summary = {
        "champion_version": float(champion.version),
        "champion_rmse": result.champion_rmse,
        "challenger_rmse": result.challenger_rmse,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "p_value": result.p_value,
    }
    record = registry.get(challenger_version)
    record.metrics.update({f"challenge_{k}": v for k, v in summary.items()})
    registry._write(record)
    if result.challenger_wins:
        registry.promote(challenger_version)
        return True, summary
    registry.promote(challenger_version, "archived")
    return False, summary
