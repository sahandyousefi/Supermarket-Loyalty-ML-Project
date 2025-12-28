# Notebooks

| Notebook | Content |
| --- | --- |
| `01_original_exam_solution.ipynb` | The original certification solution the package grew from, kept for reference |
| `02_promotion_effect.ipynb` | Does the sign-up promotion change spend: raw comparison, adjusted estimates, overlap, sensitivity, heterogeneity and the design of a randomised test |
| `03_programme_design.ipynb` | Cost and risk of the reward schedule for `test.csv`, sensitivities, behavioural response, threshold search and the effect of the noise model |

The notebooks call the package and hold no analysis code of their own; every number in them can be reproduced from the command line. They are stored with their outputs. To run them again:

```bash
pip install -e ".[notebooks]"
pytest -m slow tests/test_notebooks.py
```
