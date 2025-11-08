# Supermarket loyalty spend

This site documents a project that forecasts the annual spend of supermarket loyalty members from what is known when they join: region, tenure band, joining month, the sign-up promotion and their first month of spending. Annual spend sets the reward tier, so the forecasts feed a reward cost forecast and the design of the tier thresholds.

The project answers four questions from the original brief and then asks what a team would need before relying on the answers:

- Which attributes drive spend, and how strongly, with intervals and tests that do not assume equal variances or normal errors.
- How well spend can be forecast, comparing linear, tree, kernel and neural models under repeated cross-validation with a corrected resampled t-test.
- How uncertain each forecast is, with conformal intervals audited for coverage within subgroups and full predictive distributions.
- What the promotion does, from observational data with doubly robust estimators and sensitivity analysis, and how a randomised test of it should be designed.

## Pages

- [Methodology](methodology.md): every method, its assumptions and the results on the supplied data, with references.
- [Model card](model_card.md): intended use, data, performance and limitations of the deployed model.
- [Deployment](deployment.md): registry, champion-challenger promotion, batch scoring, the service and monitoring.
- [Language model components](llm_design.md): grounded summaries, retrieval and their evaluation.
- [API reference](api.md): the main public functions and classes, from their docstrings.

## Reproducing the results

```bash
pip install -e ".[dev,all]"
loyalty-spend report
```

The command line reference is available with `loyalty-spend --help`, and every command used in the documentation lists its options with `loyalty-spend <command> --help`.
