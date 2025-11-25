import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_release", ROOT / "scripts" / "check_release.py"
)
check_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_release)


def _copy(tmp_path):
    for name in ("pyproject.toml", "CITATION.cff", "CHANGELOG.md"):
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "src/loyalty_spend").mkdir(parents=True)
    shutil.copy(ROOT / "src/loyalty_spend/__init__.py", tmp_path / "src/loyalty_spend")
    return tmp_path


def test_repository_is_consistent():
    assert check_release.check(ROOT) == []


def test_mismatched_version_is_reported(tmp_path):
    root = _copy(tmp_path)
    cff = root / "CITATION.cff"
    cff.write_text(cff.read_text().replace("\nversion: ", "\nversion: 9", 1))
    assert any("versions differ" in p for p in check_release.check(root))


def test_release_requires_an_empty_unreleased_section(tmp_path):
    root = _copy(tmp_path)
    log = root / "CHANGELOG.md"
    text = log.read_text()
    start = text.index("## [Unreleased]") + len("## [Unreleased]")
    end = text.index("## [", start)
    log.write_text(text[:start] + "\n\n- Pending change.\n\n" + text[end:])
    assert check_release.check(root, release=True) == [
        "the unreleased section of the changelog is not empty"
    ]
    log.write_text(text[:start] + "\n\n" + text[end:])
    assert check_release.check(root, release=True) == []
