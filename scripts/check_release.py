"""Check that version numbers and release notes agree before a release.

The version must be the same in ``pyproject.toml``, the package and
``CITATION.cff``; the changelog must have a section for it whose date is
the citation's release date; and with ``--release`` the unreleased section
must be empty, so that no change is shipped without a note under its
version.

Usage::

    python scripts/check_release.py            # consistency only
    python scripts/check_release.py --release  # before tagging a release
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _match(pattern: str, text: str, what: str) -> str:
    found = re.search(pattern, text, re.MULTILINE)
    if found is None:
        raise SystemExit(f"cannot find {what}")
    return found.group(1)


def check(root: Path = ROOT, release: bool = False) -> list[str]:
    problems = []
    pyproject = _match(
        r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), "version"
    )
    package = _match(
        r'^__version__ = "([^"]+)"',
        (root / "src/loyalty_spend/__init__.py").read_text(),
        "__version__",
    )
    citation_text = (root / "CITATION.cff").read_text()
    citation = _match(r"^version: (\S+)", citation_text, "citation version")
    cited_date = _match(r"^date-released: (\S+)", citation_text, "release date")
    if len({pyproject, package, citation}) != 1:
        problems.append(
            f"versions differ: pyproject {pyproject}, package {package}, "
            f"citation {citation}"
        )
    changelog = (root / "CHANGELOG.md").read_text()
    entry = re.search(
        rf"^## \[{re.escape(pyproject)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        re.MULTILINE,
    )
    if entry is None:
        problems.append(f"no changelog section for {pyproject}")
    elif entry.group(1) != cited_date:
        problems.append(
            f"changelog date {entry.group(1)} differs from citation {cited_date}"
        )
    unreleased = re.search(
        r"^## \[Unreleased\]\n(.*?)(?=^## \[)", changelog, re.MULTILINE | re.DOTALL
    )
    if release and unreleased is not None and unreleased.group(1).strip():
        problems.append("the unreleased section of the changelog is not empty")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    problems = check(release=args.release)
    for problem in problems:
        print(problem)
    if not problems:
        print("release metadata is consistent")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
