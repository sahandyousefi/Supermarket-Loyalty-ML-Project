"""Reproducibility manifest for a run.

A result can only be reproduced with the same code, data, configuration and
library versions. The manifest records all four: the package version and
git commit, a SHA-256 hash of every input file, a hash of the resolved
configuration and the versions of the numerical libraries.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loyalty_spend import __version__

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "lightgbm")


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: Any) -> str:
    payload = (
        asdict(config)
        if is_dataclass(config) and not isinstance(config, type)
        else config
    )
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str]:
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def git_commit(cwd: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    config: Any, inputs: list[str | Path], cwd: str | Path = "."
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "git_commit": git_commit(cwd),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "config_hash": config_hash(config),
        "inputs": {str(p): file_sha256(p) for p in inputs},
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
