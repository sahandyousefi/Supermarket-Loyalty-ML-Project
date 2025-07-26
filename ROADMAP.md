# Roadmap after 1.0

Version 1.0 answers the four questions of the original brief. The work below extends the project in the directions a loyalty programme team would need next: stricter data contracts and drift monitoring, full predictive distributions instead of point forecasts, an honest estimate of what the sign-up promotion does to spend, targeting of that promotion, a reward schedule that can be designed rather than only costed, a scoring service, and grounded natural-language summaries of the results.

Items are numbered `<phase>.<item>`. Finished items move to the Completed section with the date they were finished.

## Phase 1: Engineering


## Phase 2: Data contracts and drift


## Phase 3: Probabilistic forecasting


## Phase 4: Effect of the sign-up promotion


## Phase 5: Heterogeneous effects and targeting


## Phase 6: Customer segments


## Phase 7: Reward programme design


## Phase 8: Model operations

- [ ] 8.9 Reward cost endpoint
- [ ] 8.10 Request limits and error handling
- [ ] 8.11 Request metrics endpoint
- [ ] 8.12 Monitoring job for a new scoring batch
- [ ] 8.13 Model card generated from bundle metadata
- [ ] 8.14 `loyalty-spend serve` command, Docker target and deployment notes

## Phase 9: Grounded summaries with language models

- [ ] 9.1 Language model client interface with an offline mock
- [ ] 9.2 OpenAI-compatible HTTP client as an optional extra
- [ ] 9.3 Versioned prompt templates
- [ ] 9.4 Fact sheet built from the report summary
- [ ] 9.5 Narrative summary generated only from the fact sheet
- [ ] 9.6 Numeric grounding check against the fact sheet
- [ ] 9.7 Structured output validation with a bounded retry
- [ ] 9.8 Chunking of the project documentation
- [ ] 9.9 TF-IDF retrieval with recall at k and mean reciprocal rank
- [ ] 9.10 Questions answered with citations to retrieved chunks
- [ ] 9.11 Citation verification
- [ ] 9.12 Input guardrails against prompt injection
- [ ] 9.13 Plain-language reason statements for single forecasts
- [ ] 9.14 Evaluation harness for faithfulness, format and refusals
- [ ] 9.15 Token and cost accounting
- [ ] 9.16 `loyalty-spend narrate` and `loyalty-spend ask` commands
- [ ] 9.17 Design notes for the language model components

## Phase 10: Documentation and release

- [ ] 10.1 Walkthrough notebook for the promotion analysis
- [ ] 10.2 Walkthrough notebook for reward programme design
- [ ] 10.3 Runtime benchmarks of the main commands
- [ ] 10.4 Methodology and model card updates
- [ ] 10.5 README rewrite
- [ ] 10.6 Release 2.0.0

## Completed
- [x] 2025-05-05: 1.1 Add a changelog that records the 1.0.0 release
- [x] 2025-05-05: 1.2 Add contributing notes with the development workflow
- [x] 2025-05-05: 1.3 Run the pre-commit hooks in CI as a separate job
- [x] 2025-05-05: 1.4 Enforce a coverage floor and publish the coverage report from CI
- [x] 2025-05-05: 1.5 Add mypy configuration and type the data package
- [x] 2025-05-10: 1.6 Type the configuration, evaluation and analysis modules
- [x] 2025-05-10: 1.7 Type the models, uncertainty and business modules
- [x] 2025-05-10: 1.8 Run mypy in CI
- [x] 2025-05-10: 1.9 Property-based tests for the reward schedule
- [x] 2025-05-10: 1.10 Property-based tests for numeric parsing and category normalization
- [x] 2025-05-10: 1.11 Property-based tests for the conformal quantile
- [x] 2025-05-16: 1.12 Add a `--version` flag and a `--verbose` logging flag to the CLI
- [x] 2025-05-16: 1.13 Run the slow test suite on demand from GitHub Actions
- [x] 2025-05-16: 1.14 Build and check the source distribution and wheel in CI, and ship type information
- [x] 2025-05-16: 1.15 Add a citation file
- [x] 2025-05-16: 2.1 Declarative column contracts for dtype, range and allowed levels
- [x] 2025-05-16: 2.2 Contract checks for ranges and types
- [x] 2025-05-20: 2.3 Contract checks for levels, uniqueness and missing-value budgets
- [x] 2025-05-20: 2.4 Read contracts from `configs/contracts.yaml`
- [x] 2025-05-20: 2.5 `loyalty-spend validate` command with a contract report
- [x] 2025-05-20: 2.6 Column profile: missing share, distinct count, quantiles and top levels
- [x] 2025-05-20: 2.7 Render a profile as Markdown
- [x] 2025-05-20: 2.8 Population stability index for numeric features with quantile bins
- [x] 2025-05-20: 2.9 Population stability index for categorical features, including unseen levels
- [x] 2025-05-24: 2.10 Kolmogorov-Smirnov and chi-square tests with Benjamini-Hochberg control
- [x] 2025-05-24: 2.11 Per-feature drift report combining PSI and the tests
- [x] 2025-05-24: 2.12 Multivariate drift with a domain classifier and its AUC
- [x] 2025-05-24: 2.13 Prediction drift and, when labels arrive, target drift
- [x] 2025-05-24: 2.14 Monitoring section in the configuration with drift thresholds
- [x] 2025-05-24: 2.15 `loyalty-spend drift` command
- [x] 2025-05-24: 2.16 Document the drift between `train.csv` and `test.csv`
- [x] 2025-06-02: 3.1 Proper scoring rules: pinball loss, Gaussian and sample CRPS
- [x] 2025-06-02: 3.2 Interval score, coverage and width summaries
- [x] 2025-06-02: 3.3 Probability integral transform values and a calibration test
- [x] 2025-06-02: 3.4 Linear quantile regression for several quantiles at once
- [x] 2025-06-02: 3.5 Gradient boosting quantile regression for several quantiles
- [x] 2025-06-03: 3.6 Monotone rearrangement of crossing quantiles
- [x] 2025-06-03: 3.7 Heteroscedastic Gaussian linear model with a log-variance regression
- [x] 2025-06-03: 3.8a Common quantile-grid interface for distributional models
- [x] 2025-06-03: 3.8b Cross-validated CRPS comparison of distributional models
- [x] 2025-06-03: 3.9 Mondrian conformal prediction by customer group
- [x] 2025-06-10: 3.10 Conformal predictive distributions
- [x] 2025-06-10: 3.11 Normalized conformal prediction with a difficulty model
- [x] 2025-06-10: 3.12a Weighted conformal quantile
- [x] 2025-06-10: 3.12b Weighted conformal intervals with density ratios from a domain classifier
- [x] 2025-06-10: 3.13a Side-by-side evaluation of the interval methods
- [x] 2025-06-10: 3.13b `loyalty-spend intervals` command
- [x] 2025-06-10: 3.14 Quantile reliability figure
- [x] 2025-06-15: 3.15 Draw reward cost simulations from a predictive distribution
- [x] 2025-06-15: 3.16 Report section on probabilistic evaluation
- [x] 2025-06-15: 3.17 Methodology: scoring rules and conformal extensions
- [x] 2025-06-18: 4.1 Causal module with the estimands and the identification assumptions
- [x] 2025-06-18: 4.2 Simulator option for a promotion effect with confounding
- [x] 2025-06-18: 4.3 Cross-fitted propensity model
- [x] 2025-06-18: 4.4 Overlap diagnostics and propensity trimming
- [x] 2025-06-18: 4.5 Covariate balance with standardized mean differences
- [x] 2025-06-18: 4.6 Inverse probability weighting estimator with bootstrap intervals
- [x] 2025-06-18: 4.7 Regression adjustment estimator
- [x] 2025-06-19: 4.8a Cross-fitted nuisance functions and the augmented IPW estimator of the ATE
- [x] 2025-06-19: 4.8b Augmented IPW estimator of the ATT
- [x] 2025-06-19: 4.9 Double machine learning for the partially linear model
- [x] 2025-06-19: 4.10a Partial R-squared, robustness value and omitted-variable bias of the regression estimate
- [x] 2025-06-19: 4.10b Bounds on confounding as strong as an observed covariate
- [x] 2025-06-19: 4.11 Placebo test with permuted treatment
- [x] 2025-06-19: 4.12 Estimator comparison table
- [x] 2025-06-21: 4.13 `loyalty-spend effect` command
- [x] 2025-06-21: 4.14 Figure of effect estimates and intervals
- [x] 2025-06-21: 4.15a Methodology section on the promotion effect
- [x] 2025-06-21: 4.15b README section on the promotion effect
- [x] 2025-06-21: 5.1 Simulator option for effects that vary with tenure
- [x] 2025-06-25: 5.2a S-learner and the PEHE error measure
- [x] 2025-06-25: 5.2b T-learner
- [x] 2025-06-25: 5.3 X-learner
- [x] 2025-06-25: 5.4 DR-learner
- [x] 2025-06-25: 5.5 R-learner
- [x] 2025-06-25: 5.6 Qini curve and area under the uplift curve
- [x] 2025-06-27: 5.7 Uplift by decile of predicted effect
- [x] 2025-06-27: 5.8 Best linear predictor test of effect heterogeneity
- [x] 2025-06-27: 5.9 Sorted group average treatment effects
- [x] 2025-06-27: 5.10 Targeting policy with a cost per promotion and a doubly robust policy value
- [x] 2025-06-27: 5.11 Budget-constrained targeting
- [x] 2025-06-27: 5.12 `loyalty-spend uplift` command
- [x] 2025-06-28: 5.13a Methodology and report sections on targeting
- [x] 2025-06-28: 5.13b README section on targeting
- [x] 2025-06-28: 6.1 Gower distance for mixed numeric and categorical data
- [x] 2025-06-29: 6.2 K-medoids clustering on a precomputed distance matrix
- [x] 2025-06-29: 6.3 Choice of the number of segments by average silhouette width
- [x] 2025-06-29: 6.4 Segment stability by bootstrap Jaccard similarity
- [x] 2025-06-29: 6.5 Segment profiles
- [x] 2025-06-29: 6.6 Spend forecast and reward cost by segment
- [x] 2025-06-29: 6.7a `loyalty-spend segments` command
- [x] 2025-07-04: 6.7b Segment results in the README and methodology
- [x] 2025-07-04: 7.1 Marginal tier schedules alongside whole-spend schedules
- [x] 2025-07-04: 7.2 Effective reward rate and cost-to-spend ratio
- [x] 2025-07-04: 7.3 Common random numbers for comparing schedules
- [x] 2025-07-04: 7.4 Sensitivity of expected cost to each threshold
- [x] 2025-07-14: 7.5 Threshold search under a cost budget
- [x] 2025-07-14: 7.6 Behavioural response scenario with bunching below thresholds
- [x] 2025-07-14: 7.7a Monte Carlo standard errors of the cost summaries
- [x] 2025-07-14: 7.7b Quasi-Monte Carlo draws for Gaussian predictive distributions
- [x] 2025-07-14: 7.8 Value at risk and expected shortfall of total reward cost
- [x] 2025-07-18: 7.9 Reward cost by segment and region
- [x] 2025-07-18: 7.10a Programme design analysis in one call
- [x] 2025-07-18: 7.10b `loyalty-spend design` command
- [x] 2025-07-18: 7.11a Methodology section on programme design
- [x] 2025-07-18: 7.11b Report section on programme design
- [x] 2025-07-18: 7.11c README section on programme design
- [x] 2025-07-18: 8.1 File-based model registry with versions and metadata
- [x] 2025-07-19: 8.2 Stage promotion and rollback
- [x] 2025-07-19: 8.3a Paired champion-challenger test on a holdout
- [x] 2025-07-19: 8.3b Promote a challenger through the registry only when it wins
- [x] 2025-07-19: 8.4 Reproducibility manifest with package versions and data fingerprints
- [x] 2025-07-19: 8.5a Batch scoring with intervals and contract checks
- [x] 2025-07-19: 8.5b Interval and contract options for the predict command
- [x] 2025-07-19: 8.6 Optional MLflow tracking
- [x] 2025-07-26: 8.7 FastAPI service with health and version endpoints
- [x] 2025-07-26: 8.8 Prediction endpoint with intervals
