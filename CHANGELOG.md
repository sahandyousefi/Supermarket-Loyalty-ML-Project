# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [1.5.0] - 2025-11-30

### Added

- Control charts for model monitoring: EWMA with exact limits, tabular CUSUM with Siegmund's run length approximation, HWMA and multivariate HWMA, vectorised run length simulation with limit calibration, zero- and steady-state comparisons, and the Brook-Evans Markov chain for the effect of estimated parameters; `loyalty-spend control-chart`.
- Design of a randomised promotion test: power and sample size with covariate adjustment, CUPED and Lin's interacted regression estimator, stratified randomisation, group-sequential boundaries from alpha-spending functions, and simulated operating characteristics; `loyalty-spend experiment`.
- Games-Howell pairwise comparisons, a Freedman-Lane permutation test of the partial tenure effect and noncentral F intervals for omega squared, reported in the driver section.
- Localised conformal intervals with Gower kernel weights, bandwidth selection by interval score, and an audit of cross-fitted interval coverage by subgroup; `loyalty-spend intervals --audit`.
- A Gaussian process model with a linear plus ARD kernel, restart diagnostics for the marginal likelihood and calibration of its predictive intervals.
- Error variance by joining month: tests of equal variance that keep their level without normal errors, a variance profile against attribute spread, feasible weighted least squares with log-scale shrinkage (`wls_month`), and month variances pooled through a learned prior scale in the Gibbs sampler (`bayes --pool-variances`).
- Reward cost under month-dependent noise with a scaled residual bootstrap and a comparison of noise models (`rewards --uncertainty month`, `--compare-noise`), and a month-noise scenario in the simulation study.
- Intake checks for repeated ids, duplicate and near-duplicate records and outlying amounts, with a report section.
- Reproducibility: reference values of 17 reported results, `loyalty-spend reproduce` with a record of the run environment, and a test that results do not depend on hash seeds or thread counts.
- Parquet input and output for batch scoring, request identifiers and JSON request logs in the service, a latency benchmark and a snapshot of the OpenAPI schema.
- A documentation site built with MkDocs, with an API reference generated from docstrings.
- Strict type checking for the business, evaluation, synthetic, robustness and causal packages, property-based tests of the sorted-draw costing, and a mutation testing script.
- CI jobs for the documentation, a dependency audit, the service container and the consistency of release metadata.

### Fixed

- The interacted regression estimator drops interactions and collinear columns in small samples, where its standard error could be undefined.
- Sorted-draw costing searched one flattened array with each draw shifted by an offset, which misplaced thresholds closer to a spend value than the rounding error of the offset; it now runs an exact binary search per draw.

## [1.4.0] - 2025-10-12

### Added

- Grounded summaries with language models: an OpenAI-compatible HTTP client without extra dependencies and a mock client for tests, versioned prompt templates, a fact sheet built from the report, narrative summaries checked number by number against it, structured output validation with a bounded retry, documentation chunking and retrieval, answers with verified citations, input guardrails, an evaluation harness for faithfulness, format and refusals, and token accounting; `loyalty-spend narrate` and `ask`.
- A hierarchical Bayesian model of spend: empirical Bayes shrinkage, a blocked Gibbs sampler with partially pooled region effects, Student-t errors and group variances, split R-hat and effective sample size, posterior predictive checks, WAIC against OLS, prior sensitivity and reward cost from the posterior predictive distribution; `loyalty-spend bayes`.
- Explanations: accumulated local effects, centred ICE curves, a global surrogate tree with fidelity, stability of importances across folds, and counterfactual first-month spend needed to reach the next tier; `loyalty-spend next-tier` and a service endpoint.
- Robustness checks: subgroup errors with bootstrap intervals, slice finding with false discovery rate control, stress tests on damaged inputs, multiple imputation with Rubin's rules, a delta adjustment for values missing not at random, closed-form influence diagnostics and learning curves; `loyalty-spend robustness`.
- Synthetic members: Gaussian copula, stratified and conditional synthesisers, fidelity, train-on-synthetic utility, distance to closest record and a membership inference test; `loyalty-spend synthesize`.
- Parallel fold fitting (`compare --jobs`), fold preprocessing shared across tuning trials and models, threshold search on sorted draws with prefix sums, and a memory cap for the reward simulation (`rewards --max-memory-mb`).

### Changed

- Documentation retrieval uses BM25 instead of TF-IDF; recall at 5 on the evaluation questions rises from 0.75 to 0.92.

### Fixed

- Learning curve points are grouped by training share, so folds that differ by one member no longer produce separate points.

## [1.3.0] - 2025-07-31

### Added

- Customer segments: Gower distance, k-medoids with restarts, silhouette selection, bootstrap Jaccard stability, segment profiles and cost by segment; `loyalty-spend segments`.
- Reward programme design: marginal schedules, cost-to-spend ratio and tier shares, common random numbers for schedule comparisons, cost sensitivities, threshold search under a budget, a bunching stress scenario, Monte Carlo standard errors, quasi-Monte Carlo draws, value at risk and expected shortfall, cost by region; `loyalty-spend design` and a report section.
- Model operations: a file-based registry with stages and rollback, a paired champion-challenger test, batch scoring with intervals and contract checks, a batch monitoring check, reproducibility manifests, optional MLflow tracking and generated model cards; `loyalty-spend monitor`, `model-card` and `predict --alpha --validate`.
- A FastAPI scoring service with prediction, reward cost and metrics endpoints, request validation and batch limits; `loyalty-spend serve`, a Docker serve target and deployment notes.

### Fixed

- Type errors reported by mypy with the NumPy stubs used on Python 3.10.

## [1.2.0] - 2025-06-28

### Added

- Causal analysis of the sign-up promotion: estimands and identification assumptions, cross-fitted propensity scores, overlap diagnostics with the Crump et al. trimming rule, covariate balance, Hajek IPW, regression adjustment, cross-fitted AIPW for the ATE and ATT, double machine learning for the partially linear model, a permutation placebo test and Cinelli-Hazlett sensitivity analysis with covariate benchmarks.
- Semi-synthetic promotion studies with known, optionally tenure-dependent effects and tunable confounding.
- S-, T-, X-, DR- and R-learners for conditional effects; Qini and doubly robust gain curves; best linear predictor, sorted group effects and a profile of the most and least affected members by repeated sample splitting; targeting rules with doubly robust policy values under a cost and a budget.
- `loyalty-spend effect` and `loyalty-spend uplift`, report sections and a forest plot for the promotion effect.

### Fixed

- Heterogeneity inference no longer cross-fits the CATE proxy and the scores on the same sample, which produced spurious findings.

## [1.1.0] - 2025-06-15

### Added

- Proper scoring rules: pinball loss, CRPS (closed form, sample and quantile-grid versions), interval score, PIT values and a uniformity test, quantile reliability.
- Distributional models: linear and gradient-boosting quantile regression with monotone rearrangement, a heteroscedastic Gaussian linear model and split conformal predictive distributions, behind one quantile-grid interface with cross-validated CRPS comparison.
- Mondrian, normalized and covariate-shift weighted conformal intervals, and `loyalty-spend intervals` to compare all interval methods with group coverage.
- Reward cost simulation from any predictive sampler; `loyalty-spend rewards --uncertainty heteroscedastic`.
- Probabilistic forecasting section in the report and the methodology.
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

[Unreleased]: https://github.com/sahandyousefi/Supermarket-Loyalty-ML-Project/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/sahandyousefi/Supermarket-Loyalty-ML-Project/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/sahandyousefi/Supermarket-Loyalty-ML-Project/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/sahandyousefi/Supermarket-Loyalty-ML-Project/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/sahandyousefi/Supermarket-Loyalty-ML-Project/releases/tag/v1.0.0
