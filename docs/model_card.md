# Model Card: Annual Spend Forecast

Structure follows Mitchell et al. (2019).

## Model details

- **Owner:** Sahand Yousefi
- **Version:** 2.0.0
- **Task:** regression of a loyalty member's total spend over the next full year.
- **Champion:** feasible weighted least squares with one error variance per joining month (`wls_month` in the registry): the coefficients of ordinary least squares on one-hot encoded categories and standardized numerics, refitted with weights from month variances shrunk towards the pooled variance on the log scale. Its point forecasts are those of OLS to within 0.001 in RMSE; it replaced OLS in version 2.0.0 because its intervals are calibrated within every joining month (see below).
- **Previous champion:** ordinary least squares (`ols`), kept as the reference model.
- **Challengers evaluated:** OLS, ridge, elastic net, Huber, Gamma GLM, spline ridge, Gaussian process, random forest, histogram gradient boosting, XGBoost, LightGBM, MLP with entity embeddings, MLP with periodic numeric embeddings, FT-Transformer.
- **Code:** `src/loyalty_spend/models`. Configuration: `configs/default.yaml`.

## Intended use

- Forecasting annual spend of existing members to estimate the cost of rewards and the expected profit of the programme.
- Producing prediction intervals and a distribution of total reward cost for budgeting.
- **Out of scope:** decisions about individual customers (eligibility, pricing, targeting), members whose attributes fall outside the training ranges, and forecasting after a change to the programme rules.

## Factors

- Tenure band is the dominant factor: it accounts for 88% of the explained variance in `loyalty.csv` under a Shapley decomposition of R squared.
- Region and promotion explain less than 0.1% each.
- Out-of-fold residuals have a mean within 0.001 of zero in every region, tenure band and promotion group. No group is systematically over- or under-forecast.
- Joining month is the factor that matters for uncertainty: the residual SD ranges from 0.104 (June joiners) to 0.551 (August joiners), and tests that do not assume normal errors reject equal variances across months at $p < 10^{-6}$, while finding no difference across regions or the promotion.
- No personal or protected attributes are present.

## Metrics

| Metric | Purpose |
| --- | --- |
| RMSE | Primary metric; drives the variance of the total reward forecast |
| MAE, MAPE | Typical absolute and relative error per customer |
| R squared | Share of variance explained |
| Interval coverage and width | Calibration of prediction intervals, overall, by decile of predicted spend and by joining month |
| Interval score | Width plus a penalty of $2/\alpha$ per unit of miss; the second promotion criterion |

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

**Promotion of the weighted model.** On five-fold cross-fitted forecasts for all 996 members (`loyalty-spend challenge`):

| | OLS | Weighted by month | Difference, 95% bootstrap interval |
| --- | --- | --- | --- |
| RMSE | 0.3095 | 0.3086 | -0.0009 [-0.0027, 0.0009] |
| Mean interval score, 90% intervals | 1.408 | 1.233 | -0.175 [-0.226, -0.129] |
| Coverage | 0.909 | 0.912 | |
| Joining months with coverage off target | 4 of 12 | 0 of 12 | |

The rule promotes a challenger whose RMSE is not worse than the champion's by more than 0.01 and whose interval score is better beyond sampling error; both conditions hold.

## Ethical considerations

- The forecast is used for aggregate budgeting. It should not be used to treat individual members differently, since it has not been assessed for that purpose.
- Reward schedule parameters in the configuration are illustrative; cost figures depend on them.

## Caveats and recommendations

- The data are a small synthetic sample from a certification exercise. Before production use, refit on real transaction data and repeat the comparison, including the simulation study calibrated to that data.
- Validation is cross-sectional. Validate on a later programme year before relying on the forecast across years.
- Conformal intervals assume that scored customers are exchangeable with the calibration sample. Recalibrate after any change in the membership mix.
- The month variances are estimated from 35 to 124 members per month. A cohort joining in a month with unusual first-month spread should be checked against them, and members from months not seen in training get the pooled variance.

## References

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., and Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229.
- Nadeau, C., and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
