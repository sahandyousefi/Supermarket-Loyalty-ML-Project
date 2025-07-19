import hashlib
import json

from loyalty_spend.config import Config, load_config
from loyalty_spend.ops.manifest import (
    build_manifest,
    config_hash,
    file_sha256,
    git_commit,
    write_manifest,
)


def test_file_hash_matches_hashlib(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"a,b\n1,2\n")
    assert file_sha256(path) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_config_hash_tracks_changes():
    base = load_config()
    assert config_hash(base) == config_hash(load_config())
    from dataclasses import replace

    changed = replace(base, rewards=replace(base.rewards, seed=base.rewards.seed + 1))
    assert config_hash(changed) != config_hash(base)


def test_manifest_round_trip(raw_dir, tmp_path):
    manifest = build_manifest(Config(), [raw_dir / "train.csv"])
    assert manifest["inputs"][str(raw_dir / "train.csv")] == file_sha256(
        raw_dir / "train.csv"
    )
    assert "numpy" in manifest["packages"]
    path = write_manifest(manifest, tmp_path / "out" / "manifest.json")
    assert json.loads(path.read_text())["config_hash"] == manifest["config_hash"]


def test_git_commit_outside_a_repository(tmp_path):
    assert git_commit(tmp_path) is None
