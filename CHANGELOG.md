# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Roadmap for the work after 1.0 (`ROADMAP.md`), contributing notes and a citation file.
- Declarative data contracts in `configs/contracts.yaml` and `loyalty-spend validate`.
- Column profiles with a Markdown renderer.
- Drift monitoring: PSI for numeric and categorical features, KS and chi-square tests with Benjamini-Hochberg control, a domain classifier for joint shifts, prediction and performance drift, and `loyalty-spend drift`.
- `--version` and `--verbose` flags for the command line.
- Property-based tests (Hypothesis) for the reward schedule, the cleaning rules and the conformal quantile.
- CI jobs for the pre-commit hooks, type checking with mypy, a coverage floor and a package build check; an on-demand workflow for the slow tests.

## [1.0.0] - 2025-04-27

First complete release.

### Added

- Cleaning of the raw extract to the data dictionary, with a log of every change.
- Validation of cleaned frames and a lineage check showing that the supplied files cannot be joined.
- Driver analysis: bootstrap group means, Welch's ANOVA, Tukey's HSD and the LMG decomposition of R squared.
- Model registry with linear models, tree ensembles and PyTorch networks (entity-embedding MLP, PLR embeddings, FT-Transformer, deep ensemble).
- Repeated cross-validation on shared splits and the corrected resampled t-test.
- Optuna tuning and nested cross-validation.
- Semi-synthetic simulation study calibrated to the training data.
- Split conformal, CV+ and conformalized quantile regression intervals.
- Monte Carlo forecast of total reward cost under a tiered schedule.
- Additive contributions, TreeSHAP and grouped permutation importance.
- `loyalty-spend` command line interface, results report, model card and Dockerfile.
- CI with ruff and pytest on Python 3.10 and 3.12.

[Unreleased]: https://github.com/sahandyousefi/Supermarket-Loyalty-ML-Project/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/sahandyousefi/Supermarket-Loyalty-ML-Project/releases/tag/v1.0.0
