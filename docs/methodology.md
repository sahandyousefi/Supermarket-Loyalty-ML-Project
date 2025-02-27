# Methodology

This document states the statistical design of the project: how the data are cleaned and checked, how the business question about tenure is tested, how models are compared, why a simulation study is needed, how prediction intervals are built and how reward cost is forecast. Implementation details are in the docstrings under `src/loyalty_spend`.

## 1. Problem and notation

For customer $i$ with attributes $x_i$, the target $y_i$ is total spend in the last full year of the loyalty programme. The programme pays a reward $r(y_i)$ that depends on spend through a tier schedule. The business needs two things:

1. a forecast $\hat y(x)$ for each customer, with an interval $[\ell(x), u(x)]$ that contains $y$ with a stated probability, and
2. a forecast of total reward cost $R = \sum_i r(y_i)$ for a customer file, with its distribution, for budgeting.

Throughout, $n$ is the number of training rows, $K$ the number of cross-validation folds, $J$ the number of fold scores, and $\alpha$ the miscoverage level of an interval.

## 2. Data cleaning and lineage

**Rules.** `loyalty.csv` is read as text so that placeholder values survive parsing. Placeholders (`.`, empty strings, `NA` and similar) become missing, category labels are matched to the dictionary ignoring case and surrounding whitespace, negative amounts become missing, amounts are rounded to two decimals, and `loyalty_years` becomes an ordered categorical. Every change is counted in a cleaning log. With `impute=True` the replacement values of the data dictionary are applied, which is what the brief asks for (Task 1). With `impute=False` missing values stay missing, which is what the models use: replacing a missing first-month spend by zero would tell a model that the customer spent nothing, which is a different statement from "not recorded".

**Lineage.** `train.csv` and `test.csv` contain the same customer ids as `loyalty.csv`. Before any join, the attributes that cannot differ between two extracts of the same customer (region, tenure band, joining month, promotion) are compared on the shared ids. With observed agreement $p_o$ and the agreement expected under independence $p_e = \sum_k p^{(1)}_k p^{(2)}_k$, Cohen's kappa is

$$\kappa = \frac{p_o - p_e}{1 - p_e}.$$

All four attributes have $|\kappa| < 0.06$, so the ids do not identify the same customers across files. The files are analysed separately and never joined.

## 3. Is tenure the main driver of spend? (Task 2)

The brief asks for the mean and variance of spend by tenure band. A table of means does not answer whether tenure is the *largest* driver, so four further analyses are run on `loyalty.csv`.

**Uncertainty of group means.** Percentile bootstrap intervals with 2,000 resamples per band.

**Welch's ANOVA.** Group variances differ (from 9.3 to 16.7), so the classical $F$-test is replaced by Welch's (1951) statistic. With group sizes $n_k$, means $\bar y_k$, variances $s_k^2$ and weights $w_k = n_k / s_k^2$,

$$F_W = \frac{\sum_k w_k (\bar y_k - \tilde y)^2 / (k-1)}{1 + \frac{2(k-2)}{k^2-1}\sum_k \frac{(1 - w_k/W)^2}{n_k - 1}}, \qquad \tilde y = \frac{\sum_k w_k \bar y_k}{W},\; W = \sum_k w_k,$$

referred to an $F$ distribution with $k-1$ and $(k^2-1) / \big(3\sum_k (1-w_k/W)^2/(n_k-1)\big)$ degrees of freedom.

**Effect size.** A p-value says nothing about magnitude, so $\eta^2 = SS_B/SS_T$ and the less biased $\omega^2 = (SS_B - (k-1)MS_W)/(SS_T + MS_W)$ are reported. Tukey's HSD gives simultaneous intervals for all pairwise differences.

**Relative importance.** Whether tenure is the *largest* driver is a statement about all features together. The R squared of the OLS model with all features is decomposed with the LMG method (Lindeman, Merenda and Gold, 1980; Groemping, 2006): feature $j$ receives its increase in $R^2$ averaged over all orders in which features can enter,

$$\phi_j = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(p-|S|-1)!}{p!}\big[R^2(S\cup\{j\}) - R^2(S)\big].$$

This is the Shapley value of the $R^2$ game, so the shares sum to the full-model $R^2$ and do not depend on an arbitrary order of entry. Categorical features enter as blocks of dummies. With six features, all $2^6$ subsets are fitted exactly.

## 4. Models

All models share one preprocessing layer fitted inside each training fold:

| Encoding | Numerics | Categories | Used by |
| --- | --- | --- | --- |
| one-hot | median imputation with missing indicator, standardized | one-hot, first level dropped | linear models |
| spline | as above, then a cubic B-spline basis with six knots | one-hot | additive model |
| ordinal | median imputation with indicator | integer codes, native categorical splits | tree ensembles |

| Family | Models |
| --- | --- |
| Linear | OLS (baseline), ridge with the penalty chosen by efficient leave-one-out CV, elastic net, Huber regression, Gamma GLM with log link, spline ridge (an additive model) |
| Trees | random forest, histogram gradient boosting, XGBoost (Chen and Guestrin, 2016), LightGBM (Ke et al., 2017) |
| Deep | MLP with entity embeddings (Guo and Berkhahn, 2016), MLP with periodic numeric embeddings (Gorishniy et al., 2022), FT-Transformer (Gorishniy et al., 2021), deep ensemble of Gaussian MLPs (Lakshminarayanan et al., 2017) |

**Networks.** Categories are mapped to learned embeddings; index 0 is reserved for missing and unseen levels. The MLP uses residual blocks of LayerNorm, Linear, GELU, Dropout and Linear. The PLR variant maps each numeric feature $x$ to $\mathrm{ReLU}(W[\sin(2\pi c x), \cos(2\pi c x)] + b)$ with trainable frequencies $c$; this lets a network represent sharp and oscillating effects that a plain MLP, biased towards smooth functions, fits poorly (Grinsztajn et al., 2022). The FT-Transformer turns every feature into a token, prepends a [CLS] token and reads the prediction from its final state. Training uses AdamW, a linear warmup followed by cosine decay, gradient clipping, a standardized target and early stopping on a 15% validation split, keeping the weights of the best epoch.

**Probabilistic networks.** A Gaussian head outputs a mean and a log variance and is trained with the negative log likelihood $\tfrac12[\log\sigma^2(x) + (y-\mu(x))^2/\sigma^2(x)]$. The deep ensemble averages $M$ such networks with different seeds; its predictive variance splits into an aleatoric part $\frac1M\sum_m\sigma_m^2(x)$ and an epistemic part $\frac1M\sum_m(\mu_m(x)-\bar\mu(x))^2$.

**Tuning.** Optuna's TPE sampler (Akiba et al., 2019) minimizes cross-validated RMSE. Because a score obtained while selecting hyperparameters is optimistically biased (Cawley and Talbot, 2010), nested cross-validation is available to estimate the performance of the whole tuning procedure, and the gap between inner and outer RMSE is reported as the optimism of selection.

## 5. Model comparison

Every model is evaluated on the same $5\times3$ repeated $K$-fold splits. RMSE is the primary metric because the cost forecast depends on squared errors through the variance of the total; MAE, MAPE and $R^2$ are also reported. Fold scores of different models on the same split are paired, but the folds share most of their training data, so the naive paired $t$-test is anti-conservative. Differences $d_1,\dots,d_J$ against the baseline are tested with the corrected resampled statistic of Nadeau and Bengio (2003):

$$t = \frac{\bar d}{\sqrt{\left(\frac1J + \frac{n_{\text{test}}}{n_{\text{train}}}\right)\hat\sigma_d^2}} \sim t_{J-1}.$$

## 6. Simulation study

On `train.csv` a linear model reaches an $R^2$ above 0.999, so the data cannot show when a flexible model is worth its cost: every model is close to the noise floor and the ranking reflects how well each approximates a linear function. The simulator keeps the joint distribution of the real features by resampling rows, uses the OLS fit on the real data as the base signal, and adds controlled departures:

| Scenario | Noise (share of target sd) | Heteroscedastic | Nonlinear weight | Interaction weight |
| --- | --- | --- | --- | --- |
| linear | 0.05 | no | 0 | 0 |
| noisy | 0.20 | yes | 0 | 0 |
| nonlinear | 0.05 | no | 0.25 | 0 |
| interactions | 0.05 | no | 0.25 | 0.25 |

The nonlinear terms are a saturating effect of basket size, a threshold in first-month spend and a quadratic penalty; the interactions are a promotion effect for first-year members and a tenure-by-first-month slope change. Noise is Student-$t$ with five degrees of freedom, rescaled to the target standard deviation, so occasional large deviations occur. Because the noise is known, each model's RMSE is compared with the Bayes error $\sqrt{\mathbb E[\sigma^2(x)]}$, and the excess RMSE divided by the Bayes error puts scenarios with different noise on one scale.
