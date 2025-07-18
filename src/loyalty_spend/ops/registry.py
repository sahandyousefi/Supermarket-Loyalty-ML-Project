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

    def summary(self) -> list[dict[str, Any]]:
        return [{**asdict(r), "tag": r.tag} for r in self.records()]
