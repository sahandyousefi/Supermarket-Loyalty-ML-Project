# Model Card: Annual Spend Forecast

Structure follows Mitchell et al. (2019).

## Model details

- **Owner:** Sahand Yousefi
- **Version:** 1.0.0
- **Task:** regression of a loyalty member's total spend over the next full year.
- **Champion:** ordinary least squares on one-hot encoded categories and standardized numerics (`ols` in the registry). Ridge regression is statistically indistinguishable and can replace it when more features are added.
- **Challengers evaluated:** ridge, elastic net, Huber, Gamma GLM, spline ridge, random forest, histogram gradient boosting, XGBoost, LightGBM, MLP with entity embeddings, MLP with periodic numeric embeddings, FT-Transformer.
- **Code:** `src/loyalty_spend/models`. Configuration: `configs/default.yaml`.

## Intended use

- Forecasting annual spend of existing members to estimate the cost of rewards and the expected profit of the programme.
- Producing prediction intervals and a distribution of total reward cost for budgeting.
- **Out of scope:** decisions about individual customers (eligibility, pricing, targeting), members whose attributes fall outside the training ranges, and forecasting after a change to the programme rules.

## Factors

- Tenure band is the dominant factor: it accounts for 88% of the explained variance in `loyalty.csv` under a Shapley decomposition of R squared.
- Region and promotion explain less than 0.1% each.
- Out-of-fold residuals of the champion have a mean within 0.001 of zero in every region, tenure band and promotion group, with standard deviations between 0.24 (10+ years) and 0.39 (1-3 years). No group is systematically over- or under-forecast.
- No personal or protected attributes are present.

## Metrics

| Metric | Purpose |
| --- | --- |
| RMSE | Primary metric; drives the variance of the total reward forecast |
| MAE, MAPE | Typical absolute and relative error per customer |
| R squared | Share of variance explained |
| Interval coverage and width | Calibration of prediction intervals, overall and by decile of predicted spend |

## Evaluation data

Five-fold cross-validation repeated three times on the 996 rows of `train.csv`, with identical splits for every model. Differences between models are tested with the corrected resampled t-test of Nadeau and Bengio (2003).

## Quantitative analysis

| Model | RMSE | MAE | R squared |
| --- | --- | --- | --- |
| OLS (champion) | 0.308 (0.040) | 0.215 | 0.9999 |
| Histogram gradient boosting | 0.433 (0.053) | 0.297 | 0.9997 |
| MLP with periodic embeddings | 0.454 (0.037) | 0.341 | 0.9997 |
| FT-Transformer | 0.555 (0.047) | 0.424 | 0.9996 |

Standard deviations across folds in parentheses. Every non-linear model is significantly worse than OLS (corrected p below 0.001). The training data are nearly noise free; the simulation study in the README shows the expected ranking when they are not.

## Ethical considerations

- The forecast is used for aggregate budgeting. It should not be used to treat individual members differently, since it has not been assessed for that purpose.
- Reward schedule parameters in the configuration are illustrative; cost figures depend on them.

## Caveats and recommendations

- The data are a small synthetic sample from a certification exercise. Before production use, refit on real transaction data and repeat the comparison, including the simulation study calibrated to that data.
- Validation is cross-sectional. Validate on a later programme year before relying on the forecast across years.
- Conformal intervals assume that scored customers are exchangeable with the calibration sample. Recalibrate after any change in the membership mix.

## References

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., and Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229.
- Nadeau, C., and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
