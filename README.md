# Supermarket Loyalty: Annual Spend Forecasting

International Essentials runs a loyalty programme in which customers earn rewards that grow with their annual spend. To budget for the cost of those rewards, the business needs a forecast of how much each member will spend over the year.

This repository started as a solution to a certification exercise and is being rebuilt as a tested Python package. The original notebook is kept in `notebooks/` as a record of the first attempt.

## Status

Work in progress. Implemented so far:

- Data dictionary as code, cleaning with a change log, and a lineage check between the supplied files
- Statistical analysis of spend by tenure: bootstrap intervals, Welch's ANOVA, effect sizes, Tukey's HSD and a Shapley decomposition of R squared
- A model registry with linear models (OLS, ridge, elastic net, Huber, Gamma GLM, spline ridge) and tree ensembles (random forest, histogram gradient boosting, XGBoost, LightGBM)
- Repeated cross-validation on fixed splits with the corrected resampled t-test of Nadeau and Bengio (2003)
- A simulator calibrated to the training data, used to compare model families when the true noise level is known

Planned: neural network models, prediction intervals and a forecast of reward cost.

## Quick start

```bash
pip install -e ".[dev,gbm]"
loyalty-spend clean --output clean.csv
loyalty-spend drivers
loyalty-spend compare --models ols ridge hist_gbm lightgbm
```

## Data

See [data/README.md](data/README.md) for the files, the data dictionary and the quality issues found in `loyalty.csv`.

## Development

```bash
pytest -q
ruff check .
```
