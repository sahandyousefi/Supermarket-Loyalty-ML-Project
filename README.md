# Supermarket Loyalty: Annual Spend Forecasting

International Essentials runs a loyalty programme in which customers earn rewards that grow with their annual spend. To budget for the cost of those rewards, the business needs a forecast of how much each member will spend over the year.

This repository started as a solution to a certification exercise and is being rebuilt as a tested Python package. The original notebook is kept in `notebooks/` as a record of the first attempt.

## Status

Work in progress. Implemented so far:

- Data dictionary as code, with validation of column types, category levels and identifiers
- Cleaning of the raw extract with a log of every change (Task 1 of the brief)
- A lineage check showing that `loyalty.csv` and `train.csv` share customer ids but not customer attributes, so the files must not be joined

Planned: the analysis of spend by tenure (Task 2), baseline and comparison models (Tasks 3 and 4), and a forecast of reward cost.

## Data

See [data/README.md](data/README.md) for the files, the data dictionary and the quality issues found in `loyalty.csv`.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```
