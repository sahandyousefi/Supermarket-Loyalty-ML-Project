"""Mutation testing of one module against a chosen set of tests.

Each mutant changes one operator or constant in the module: a comparison
is replaced by its neighbour (``<`` by ``<=``, ``>=`` by ``>``), an
arithmetic operator by another (``+`` by ``-``, ``*`` by ``/``), a numeric
constant is changed by one, and ``max``/``min`` and ``np.maximum``/
``np.minimum`` are swapped. The mutated module is written to a copy of the
source tree, which is put first on ``PYTHONPATH``, and the tests run once
per mutant. A mutant that the tests do not detect points at behaviour the
tests do not pin down.

Usage::

    python scripts/mutation_check.py src/loyalty_spend/business/rewards.py \\
        tests/test_rewards.py tests/test_properties_rewards.py
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

COMPARE = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
BINARY = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}
SWAP_NAMES = {"max": "min", "min": "max", "maximum": "minimum", "minimum": "maximum"}


def _sites(tree: ast.AST) -> Iterator[tuple[ast.AST, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in COMPARE:
            yield node, "compare"
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            yield node, "binary"
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            yield node, "constant"
        elif isinstance(node, ast.Name) and node.id in SWAP_NAMES:
            yield node, "name"
        elif isinstance(node, ast.Attribute) and node.attr in SWAP_NAMES:
            yield node, "attribute"


def _skipped(tree: ast.AST) -> set[int]:
    """Nodes in annotations and default argument values.

    Defaults are settings, such as the number of simulations; changing them
    alters what a caller gets without the defaults but is not a fault in
    the logic, so they are left out.
    """
    skip: set[int] = set()
    for node in ast.walk(tree):
        for field in ("annotation", "returns"):
            sub = getattr(node, field, None)
            if sub is not None:
                skip.update(id(n) for n in ast.walk(sub))
        if isinstance(node, ast.arguments):
            for default in [*node.defaults, *node.kw_defaults]:
                if default is not None:
                    skip.update(id(n) for n in ast.walk(default))
    return skip


def mutants(source: str) -> Iterator[tuple[int, str, str]]:
    """Yield (line, description, mutated source) for every mutation site."""
    n_sites = sum(1 for _ in _sites(ast.parse(source)))
    for k in range(n_sites):
        tree = ast.parse(source)
        skip = _skipped(tree)
        node, kind = list(_sites(tree))[k]
        if id(node) in skip:
            continue
        line = getattr(node, "lineno", 0)
        if kind == "compare":
            old = type(node.ops[0])  # type: ignore[attr-defined]
            node.ops[0] = COMPARE[old]()  # type: ignore[attr-defined]
            desc = f"{old.__name__} -> {COMPARE[old].__name__}"
        elif kind == "binary":
            old = type(node.op)  # type: ignore[attr-defined]
            node.op = BINARY[old]()  # type: ignore[attr-defined]
            desc = f"{old.__name__} -> {BINARY[old].__name__}"
        elif kind == "constant":
            value = node.value  # type: ignore[attr-defined]
            node.value = value + 1  # type: ignore[attr-defined]
            desc = f"{value!r} -> {value + 1!r}"
        elif kind == "name":
            old_name = node.id  # type: ignore[attr-defined]
            node.id = SWAP_NAMES[old_name]  # type: ignore[attr-defined]
            desc = f"{old_name} -> {node.id}"  # type: ignore[attr-defined]
        else:
            old_name = node.attr  # type: ignore[attr-defined]
            node.attr = SWAP_NAMES[old_name]  # type: ignore[attr-defined]
            desc = f"{old_name} -> {node.attr}"  # type: ignore[attr-defined]
        yield line, desc, ast.unparse(tree)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("module", type=Path)
    parser.add_argument("tests", nargs="+")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    root = Path.cwd()
    source = args.module.read_text()
    rel = args.module.resolve().relative_to(root / "src")
    killed, survived = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(root / "src", Path(tmp) / "src")
        target = Path(tmp) / "src" / rel
        env = {**os.environ, "PYTHONPATH": str(Path(tmp) / "src")}
        command = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"]
        for line, desc, mutated in mutants(source):
            target.write_text(mutated)
            try:
                result = subprocess.run(
                    [*command, *args.tests],
                    env=env,
                    capture_output=True,
                    timeout=args.timeout,
                )
                caught = result.returncode != 0
            except subprocess.TimeoutExpired:
                caught = True
            if caught:
                killed += 1
            else:
                survived.append((line, desc))
            print(f"line {line:4d}  {desc:22s} {'killed' if caught else 'SURVIVED'}")
        target.write_text(source)
    total = killed + len(survived)
    print(f"\n{killed} of {total} mutants killed ({killed / max(total, 1):.0%})")
    for line, desc in survived:
        print(f"survived: line {line}, {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
