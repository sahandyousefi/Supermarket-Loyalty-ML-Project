# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Roadmap for the work after 1.0 (`ROADMAP.md`).

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
