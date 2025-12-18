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


## Phase 9: Grounded summaries with language models


## Phase 10: Hierarchical Bayesian model of spend

The point models treat region and tenure as fixed effects. A hierarchical model shares strength between groups, gives full posterior uncertainty for every quantity, and adds parameter uncertainty to the reward cost forecast.


## Phase 11: Explanations and counterfactuals


## Phase 12: Robustness and error analysis


## Phase 13: Synthetic data with privacy checks


## Phase 14: Performance


## Phase 15: Documentation and release

These items close the year and come after phases 16 to 32.

- [ ] 15.1 Walkthrough notebook for the promotion analysis
- [ ] 15.2 Walkthrough notebook for reward programme design
- [ ] 15.3 Runtime benchmarks of the main commands
- [ ] 15.4 Methodology and model card updates
- [ ] 15.5 README rewrite
- [ ] 15.6 Release 2.0.0

## Phase 16: Control charts for model monitoring

The batch monitor of phase 8 compares one batch with the training data. Errors that creep up slowly over many batches are better caught by control charts on the standardised residuals of scored members once their spend is known.


## Phase 17: Design of the next promotion experiment

Phase 4 found no detectable effect of the sign-up promotion in observational data. A randomised test is the way to settle it.


## Phase 18: Group comparisons in the driver analysis


## Phase 19: Conditional coverage of prediction intervals


## Phase 20: Gaussian process regression


## Phase 21: Documentation site


## Phase 22: Code quality


## Phase 23: Serving


## Phase 24: Error variance that depends on the joining month

The coverage audit of phase 19 shows that forecast errors are larger for members who joined in August and September and smaller for April, June and October joiners. The model treats the noise as constant.


## Phase 25: Reproducibility


## Phase 26: Checks at intake


## Phase 27: Release hygiene


## Phase 28: Reward cost with month-dependent noise

The residual bootstrap of phase 3 draws every member's error from the same pool, although errors are five times larger for some joining months than for others.


## Phase 29: Simulation scenarios with month-dependent noise


## Phase 30: Champion choice on interval quality


## Phase 31: Error scale of a new joining cohort

Across joining months the error SD follows the within-month SD of first-month spend (phase 24), which is known a month after a cohort joins.


## Phase 32: Warnings

- [ ] 32.1 License metadata as an SPDX expression
- [ ] 32.2 Influence diagnostics for members with a hat value of one
- [ ] 32.3 Expected warnings handled where they arise
- [ ] 32.4 Unexpected warnings fail the tests

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
- [x] 2025-07-26: 8.9 Reward cost endpoint
- [x] 2025-07-26: 8.10 Request limits and error handling
- [x] 2025-07-26: 8.11 Request metrics endpoint
- [x] 2025-07-26: 8.12a Monitoring check of a new scoring batch
- [x] 2025-07-26: 8.12b `loyalty-spend monitor` command
- [x] 2025-07-31: 8.13 Model card generated from bundle metadata
- [x] 2025-07-31: 8.14a `loyalty-spend serve` and `loyalty-spend model-card` commands
- [x] 2025-07-31: 8.14b Docker target for the scoring service
- [x] 2025-07-31: 8.14c Deployment notes
- [x] 2025-07-31: 9.1 Language model client interface with an offline mock
- [x] 2025-08-07: 9.2 OpenAI-compatible HTTP client without extra dependencies
- [x] 2025-08-07: 9.3 Versioned prompt templates
- [x] 2025-08-07: 9.4 Fact sheet built from the report summary
- [x] 2025-08-07: 9.5 Narrative summary generated only from the fact sheet
- [x] 2025-08-07: 9.6 Numeric grounding check against the fact sheet
- [x] 2025-08-07: 9.7 Structured output validation with a bounded retry
- [x] 2025-08-17: 9.8 Chunking of the project documentation
- [x] 2025-08-17: 9.9 TF-IDF retrieval with recall at k and mean reciprocal rank
- [x] 2025-08-17: 9.10 Questions answered with citations to retrieved chunks
- [x] 2025-08-17: 9.11 Citation verification
- [x] 2025-08-17: 9.12 Input guardrails against prompt injection
- [x] 2025-08-17: 9.13 Plain-language reason statements for single forecasts
- [x] 2025-08-20: 9.14a Evaluation harness for faithfulness, format and refusals
- [x] 2025-08-20: 9.14b Question set with answerable and unanswerable questions
- [x] 2025-08-20: 9.15 Token and cost accounting
- [x] 2025-08-20: 9.16a `loyalty-spend narrate` command
- [x] 2025-08-20: 9.16b `loyalty-spend ask` command
- [x] 2025-08-28: 9.17 Design notes for the language model components
- [x] 2025-08-28: 10.1 Empirical Bayes shrinkage of group means
- [x] 2025-08-28: 10.2 Gibbs sampler for a hierarchical normal model with group effects
- [x] 2025-08-30: 10.3 Split R-hat and effective sample size diagnostics
- [x] 2025-08-30: 10.4 Posterior predictive draws for new members
- [x] 2025-08-30: 10.5 Posterior predictive checks
- [x] 2025-08-30: 10.6 Student-t errors as a scale mixture of normals
- [x] 2025-08-30: 10.7 Group-specific residual variances
- [x] 2025-08-30: 10.8 WAIC and cross-validated log score against OLS
- [x] 2025-08-30: 10.9 Prior sensitivity analysis
- [x] 2025-09-09: 10.10 Reward cost from the posterior predictive distribution
- [x] 2025-09-09: 10.11 `loyalty-spend bayes` command
- [x] 2025-09-09: 10.12a Methodology section on the hierarchical model
- [x] 2025-09-09: 10.12b Report section and group contrasts
- [x] 2025-09-09: 10.12c README section on the hierarchical model
- [x] 2025-09-09: 11.1 Accumulated local effects for numeric features
- [x] 2025-09-10: 11.2 Partial dependence with centred ICE curves
- [x] 2025-09-10: 11.3 Figures for ALE and ICE
- [x] 2025-09-10: 11.4 Counterfactual spend needed to reach the next tier
- [x] 2025-09-10: 11.5 Counterfactual search with immutable attributes and plausibility bounds
- [x] 2025-09-10: 11.6a `loyalty-spend next-tier` command
- [x] 2025-09-10: 11.6b Next-tier endpoint in the service
- [x] 2025-09-13: 11.7 Global surrogate tree with fidelity
- [x] 2025-09-13: 11.8 Stability of attributions across folds
- [x] 2025-09-13: 11.9 Documentation of the explanation methods
- [x] 2025-09-13: 12.1 Subgroup errors with bootstrap intervals
- [x] 2025-09-13: 12.2 Slice finding over pairs of attributes
- [x] 2025-09-13: 12.3 Stress tests with perturbed and missing inputs
- [x] 2025-09-15: 12.4a Multiple imputation with chained equations
- [x] 2025-09-15: 12.4b Rubin's rules for pooled regression estimates
- [x] 2025-09-15: 12.5 Missing-not-at-random scenarios
- [x] 2025-09-15: 12.6 Influence diagnostics: hat values, Cook's distance, DFBETAS
- [x] 2025-09-15: 12.7 Learning curves with intervals
- [x] 2025-09-15: 12.8a `loyalty-spend robustness` command
- [x] 2025-09-17: 12.8b Robustness results in the methodology and README
- [x] 2025-09-17: 13.1 Gaussian copula synthesiser for mixed data
- [x] 2025-09-17: 13.2 Fidelity: marginal and pairwise dependence comparisons
- [x] 2025-09-30: 13.1b Amounts conditional on the categories, residuals from a copula
- [x] 2025-09-30: 13.3 Utility: train on synthetic, test on real
- [x] 2025-09-30: 13.4 Privacy: distance to closest record and nearest-neighbour distance ratio
- [x] 2025-09-30: 13.5 Membership inference test
- [x] 2025-10-06: 13.6 `loyalty-spend synthesize --copula` and documentation
- [x] 2025-10-06: 14.1 Profile and vectorise the reward simulation
- [x] 2025-10-06: 14.2 Parallel cross-validation
- [x] 2025-10-06: 14.3 Cached preprocessing in repeated cross-validation
- [x] 2025-10-06: 14.4 Memory-bounded simulation for large customer files
- [x] 2025-10-12: 16.1 EWMA chart with exact time-varying control limits
- [x] 2025-10-12: 16.2 Tabular CUSUM with the decision interval set by in-control run length
- [x] 2025-10-12: 16.3 HWMA and MHWMA charts
- [x] 2025-10-12: 16.4 Average run length by Monte Carlo with standard errors
- [x] 2025-10-16: 16.5 Out-of-control run lengths under mean and variance shifts
- [x] 2025-10-16: 16.6 Effect of estimated in-control parameters on run length
- [x] 2025-10-16: 16.7 `loyalty-spend control-chart` on a stream of scored batches
- [x] 2025-10-16: 16.8 Documentation
- [x] 2025-10-17: 17.1 Sample size and power for a difference in mean spend
- [x] 2025-10-17: 17.2 Variance reduction with first-month spend (CUPED)
- [x] 2025-10-17: 17.3 Stratified randomisation by tenure band and region
- [x] 2025-10-17: 17.4 Group-sequential design with an alpha-spending function
- [x] 2025-10-17: 17.5 Operating characteristics by simulation
- [x] 2025-10-21: 17.6 `loyalty-spend experiment` command
- [x] 2025-10-21: 17.7 Documentation
- [x] 2025-10-21: 18.1 Games-Howell pairwise comparisons for unequal variances
- [x] 2025-10-30: 18.2 Permutation test of the tenure effect
- [x] 2025-10-30: 18.3 Confidence intervals for omega squared
- [x] 2025-10-30: 18.4 Report section with the pairwise comparisons
- [x] 2025-10-30: 19.1 Coverage audit by subgroup with exact binomial intervals
- [x] 2025-11-01: 19.2 Localised conformal intervals with kernel weights
- [x] 2025-11-01: 19.3 Bandwidth of the localised intervals by cross-validated interval score
- [x] 2025-11-01: 19.4 Comparison of interval methods by conditional coverage
- [x] 2025-11-01: 19.5 Documentation
- [x] 2025-11-02: 20.1 Kernel for mixed amounts and categories
- [x] 2025-11-02: 20.2 Hyperparameters by marginal likelihood with restarts
- [x] 2025-11-02: 20.3 Predictive intervals and their calibration
- [x] 2025-11-02: 20.4 Registration and comparison with the other models
- [x] 2025-11-08: 21.1 MkDocs configuration with the existing documents
- [x] 2025-11-08: 21.2 API reference from docstrings
- [x] 2025-11-08: 21.3 Strict documentation build in CI
- [x] 2025-11-08: 22.1 Strict type checking for `business`
- [x] 2025-11-08: 22.2 Strict type checking for `evaluation`
- [x] 2025-11-08: 22.3 Strict type checking for `synthetic` and `robustness`
- [x] 2025-11-10: 22.4 Strict type checking for `causal`
- [x] 2025-11-10: 22.5 Property-based tests for the sorted-draw costing
- [x] 2025-11-10: 22.6 Mutation testing of the reward module
- [x] 2025-11-10: 23.1 Batch scoring from Parquet files
- [x] 2025-11-10: 23.2 Request identifiers and structured logs
- [x] 2025-11-19: 23.3 Latency benchmark of the service
- [x] 2025-11-19: 23.4 Snapshot test of the OpenAPI schema
- [x] 2025-11-19: 24.1 Tests of equal error variance across joining months
- [x] 2025-11-19: 24.2 Feasible weighted least squares with a variance per month
- [x] 2025-11-20: 24.3 Predictive intervals and calibration of the weighted model
- [x] 2025-11-20: 24.4 Month-specific variances in the hierarchical Bayesian model
- [x] 2025-11-20: 24.5 Residual diagnostics of the high-variance months
- [x] 2025-11-20: 24.6 Documentation
- [x] 2025-11-24: 25.1 Reference values of the reported results with tolerances
- [x] 2025-11-24: 25.2 `loyalty-spend reproduce` to compare a fresh run with the reference
- [x] 2025-11-24: 25.3 Determinism across processes and thread counts
- [x] 2025-11-24: 25.4 Record of the run environment with the reproduced results
- [x] 2025-11-24: 26.1 Duplicate and near-duplicate members
- [x] 2025-11-25: 26.2 Robust outlier screening of the amounts
- [x] 2025-11-25: 26.3 Intake section in the report
- [x] 2025-11-25: 27.1 Consistency check of version, changelog and citation before a release
- [x] 2025-11-25: 27.2 Dependency audit in CI
- [x] 2025-11-29: 27.3 Container build and smoke test in CI
- [x] 2025-11-29: 28.1 Sampler from the weighted model with a per-member error SD
- [x] 2025-11-29: 28.2 Cost, tail risk and tier shares under month-dependent noise
- [x] 2025-11-29: 28.3 `rewards --uncertainty month`
- [x] 2025-11-30: 28.4 Documentation
- [x] 2025-11-30: 29.1 Scenario whose noise differs by joining month
- [x] 2025-11-30: 29.2 Interval methods under that scenario
- [x] 2025-11-30: 29.3 Weighted model in the simulation study
- [x] 2025-12-17: 30.1 Champion-challenger rule that weighs the interval score
- [x] 2025-12-17: 30.2 Decision between OLS and the weighted model
- [x] 2025-12-17: 30.3 Model card for the champion
- [x] 2025-12-18: 31.1 Error scale of a cohort predicted from its first-month spread
- [x] 2025-12-18: 31.2 Intervals for joining months not seen in training
- [x] 2025-12-18: 31.3 Leave-one-month-out evaluation
