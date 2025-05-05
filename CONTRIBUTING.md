# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, for the deep models
pip install -e ".[dev,all]"
pre-commit install
```

## Workflow

1. Create a branch from `main`.
2. Make one focused change at a time and add tests with it. Tests that train networks or run searches are marked `slow`.
3. Run the checks before pushing:

   ```bash
   make lint
   make test        # fast suite, as in CI
   make test-all    # includes the slow tests
   ```

4. Update `CHANGELOG.md` under *Unreleased* and, if the change finishes a roadmap item, move it to the Completed section of `ROADMAP.md`.

## Conventions

- Python 3.10 or later, type hints on public functions, NumPy-style docstrings.
- Formatting and linting with ruff (line length 88). The pre-commit hooks apply both.
- New models register themselves with `loyalty_spend.models.registry.register`, so the CLI, the comparison and the tuner pick them up without further changes.
- Optional dependencies go into an extra in `pyproject.toml`. Tests that need them call `pytest.importorskip`.
- Randomness is controlled by explicit seeds. A test that depends on a random draw fixes the seed and checks a property with a tolerance, not an exact value.
- Results reported in the README come from `loyalty-spend report`, never from hand-edited numbers.

## Commit messages

Imperative mood, a subject of at most 72 characters, and a body when the reason for the change is not obvious from the diff.
