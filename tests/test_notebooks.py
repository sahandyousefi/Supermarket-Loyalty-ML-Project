"""The walkthrough notebooks must run from top to bottom without errors."""

from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["02_promotion_effect", "03_programme_design"])
def test_notebook_runs(name):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(NOTEBOOKS / f"{name}.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=900, resources={"metadata": {"path": str(NOTEBOOKS)}}
    )
    client.execute()
    errors = [
        o
        for cell in nb.cells
        if cell.cell_type == "code"
        for o in cell.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors
