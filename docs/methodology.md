# Methodology

This document states the statistical design of the project: how the data are cleaned and checked, how the business question about tenure is tested, how models are compared, why a simulation study is needed, how prediction intervals and full predictive distributions are built and scored, and how reward cost is forecast. Implementation details are in the docstrings under `src/loyalty_spend`.

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

## 7. Prediction intervals

Conformal prediction gives intervals with finite-sample marginal coverage $\Pr(y \in [\ell(x), u(x)]) \ge 1-\alpha$ for any model, assuming only exchangeability.

**Split conformal** (Lei et al., 2018). Fit on one part of the data, compute absolute residuals $s_i = |y_i - \hat y(x_i)|$ on a calibration part of size $m$, and use the $\lceil (m+1)(1-\alpha)\rceil$-th smallest score $q$ as a constant half-width.

**CV+** (Barber et al., 2021). Uses every row for both fitting and calibration by combining out-of-fold residuals with the fold models' predictions; coverage is guaranteed at $1 - 2\alpha - \sqrt{2/n}$ and is close to $1-\alpha$ in practice.

**Conformalized quantile regression** (Romano et al., 2019). Gradient-boosted models estimate the $\alpha/2$ and $1-\alpha/2$ quantiles, and the conformity score $\max(\hat q_{\text{lo}}(x) - y, y - \hat q_{\text{hi}}(x))$ calibrates the band. The width adapts to each customer, which matters when the noise is heteroscedastic: constant-width intervals then over-cover low spenders and under-cover high spenders, even though the marginal coverage is correct. Coverage is therefore also reported by decile of predicted spend.

**Group-conditional coverage.** Marginal coverage can hide systematic failure for a subgroup. *Mondrian* conformal prediction (Vovk et al., 2005) computes the calibration quantile separately within groups $g$, which gives $\Pr(y \in C(x) \mid G(x) = g) \ge 1-\alpha$ for every group. Groups are either a feature such as the tenure band or quantile bins of the point forecast; groups with fewer than 30 calibration points fall back to the pooled quantile, because the conformal quantile of a very small group is infinite.

**Normalized scores.** With the score $s_i = |y_i - \hat y(x_i)| / (\hat\sigma(x_i) + \beta)$ the half-width becomes $\hat q\,(\hat\sigma(x) + \beta)$ and widens where errors are large (Papadopoulos et al., 2008). The difficulty model $\hat\sigma$ is gradient boosting with the absolute loss, fitted to out-of-fold absolute residuals of the proper training set; in-sample residuals of a flexible model would understate the difficulty. Marginal coverage holds for any $\hat\sigma$; a good one makes coverage more even across customers.

**Covariate shift.** If the customers to be scored differ in their features but not in how spend depends on the features, exchangeability fails. Weighted conformal prediction (Tibshirani et al., 2019) restores coverage by giving calibration point $i$ the weight $w(X_i) = dP_{\text{new}}/dP_{\text{train}}(X_i)$ and using the $1-\alpha$ quantile of $\sum_i p_i(x)\delta_{s_i} + p_{m+1}(x)\delta_{\infty}$ with $p_i(x) = w(X_i) / (\sum_j w(X_j) + w(x))$. The likelihood ratio is estimated from the domain classifier of Section 12 as $\frac{\hat p(x)}{1-\hat p(x)}\cdot\frac{m}{n_{\text{new}}}$, clipped to $[1/20, 20]$ because a few extreme weights would otherwise make the intervals infinite. Estimation error in the weights makes the intervals slightly conservative in the tests.

## 8. Probabilistic forecasts

An interval answers one question at one level. Budgeting reward cost needs the whole predictive distribution of each customer's spend, and the distributions have to be judged as distributions.

**Scoring rules.** A scoring rule $S(F, y)$ is proper if the expected score under the true distribution $G$ is minimised by $F = G$ (Gneiting and Raftery, 2007), so a forecaster cannot gain by misreporting. Three proper rules are used. The pinball loss of a $\tau$ quantile forecast $q$ is

$$\rho_\tau(y - q) = \max\{\tau (y-q),\; (\tau - 1)(y-q)\}.$$

The continuous ranked probability score

$$\mathrm{CRPS}(F, y) = \int \big(F(z) - \mathbb 1\{z \ge y\}\big)^2 dz = \mathbb E_F|X-y| - \tfrac12 \mathbb E_F|X-X'|$$

is in the units of spend and reduces to the absolute error for a point forecast. It equals $2\int_0^1 \rho_\tau(y - F^{-1}(\tau))\,d\tau$, so every model is scored on a common grid of 19 levels $\tau \in \{0.05, 0.10, \dots, 0.95\}$, which treats quantile, Gaussian and conformal forecasts alike. For a normal forecast the closed form of Gneiting et al. (2005) is used in the tests; for samples, the sorted-sample identity $\mathbb E|X-X'| = \tfrac{2}{m^2}\sum_i (2i-m-1)x_{(i)}$ avoids the quadratic pairwise sum. The interval score of a central $1-\alpha$ interval,

$$\mathrm{IS}_\alpha(\ell, u; y) = (u - \ell) + \tfrac{2}{\alpha}(\ell - y)\mathbb 1\{y < \ell\} + \tfrac{2}{\alpha}(y - u)\mathbb 1\{y > u\},$$

rewards narrow intervals only as long as they keep their coverage.

**Calibration.** If $F$ is the true conditional distribution, the probability integral transform $u = F(y)$ is uniform (Dawid, 1984). PIT values are tested for uniformity with a chi-square test over ten bins; a U-shape means intervals that are too narrow, a hump means intervals that are too wide. Quantile reliability plots the share of outcomes below each forecast quantile against its level.

**Models.**

- *Gaussian with constant variance:* OLS with the residual standard deviation, the reference model.
- *Heteroscedastic Gaussian:* $y \mid x \sim \mathcal N(x^\top\beta, \exp(z^\top\gamma))$, fitted in two stages. The log of a squared Gaussian residual is biased by $\mathbb E[\log\chi^2_1] \approx -1.27$, so the variance is rescaled until the standardized training residuals have unit mean square.
- *Quantile regression:* one linear programme per level (Koenker and Bassett, 1978) or gradient boosting with the pinball loss. Separately fitted quantiles can cross; each row is sorted, which is the monotone rearrangement of Chernozhukov et al. (2010) and never increases the distance to the true quantile curve.
- *Conformal predictive distribution:* the distribution of $\hat y(x) + R$, where $R$ puts mass $1/(m+1)$ on each of the $m$ signed calibration residuals (Vovk et al., 2019). Its PIT is uniform up to $1/(m+1)$ under exchangeability, whatever the point model.

Differences in cross-validated CRPS are tested with the corrected resampled $t$-test of Section 5.

## 9. Reward cost forecast

The reward of a customer is $r(y) = \rho_{t(y)}\, y$, where $t(y)$ is the highest tier whose threshold $y$ reaches and $\rho_t$ its rate. Because $r$ jumps at every threshold, $\mathbb E[r(Y)] \ne r(\mathbb E[Y])$ in general, and applying the schedule to point forecasts misstates cost for customers near a threshold. The forecast propagates uncertainty by simulation: for $b = 1,\dots,B$ draws,

$$y_i^{(b)} = \hat y(x_i) + e^{(b)}_i, \qquad R^{(b)} = \sum_i r\big(y_i^{(b)}\big),$$

where $e^{(b)}_i$ are resampled out-of-fold residuals of the model. The distribution of $R^{(b)}$ gives the expected total and its 5th and 95th percentiles; the share of draws in which a customer's tier differs from the tier of the point forecast identifies the customers that drive cost uncertainty. Residual resampling assumes that errors do not depend on $x$. Under heteroscedastic noise, draws come from a model of the conditional distribution instead: `loyalty-spend rewards --uncertainty heteroscedastic` uses the heteroscedastic Gaussian model of Section 8, and any other predictive sampler can be passed to the simulation.

### 9.1 Designing the schedule

Costing the current schedule is one question; choosing a schedule is another.

**Schedule types.** Under a *whole-spend* schedule the rate of the highest tier reached applies to all of a member's spend, so the reward jumps at every threshold. Under a *marginal* schedule each rate applies only to the spend inside its band, like income tax brackets, and the reward is continuous. With the same rates a marginal schedule is far cheaper; the comparison of interest is between schedules tuned to the same budget.

**Common random numbers.** All schedules are costed on the same draws of spend. The Monte Carlo variance of a difference is $\mathrm{Var}(A) + \mathrm{Var}(B) - 2\,\mathrm{Cov}(A, B)$, and the covariance is large because every schedule pays more when spend is high, so differences between similar schedules are resolved with a fraction of the draws that independent simulations would need.

**Sensitivities.** The derivative of expected cost with respect to each threshold $t_k$ and rate $\rho_k$ is estimated by central differences on common random numbers. For a whole-spend schedule, $\partial\,\mathbb E[R]/\partial \rho_k$ is the expected spend of members in tier $k$, and $\partial\,\mathbb E[R]/\partial t_k$ is approximately $-f_S(t_k)\,(\rho_k - \rho_{k-1})\,t_k$ summed over members, large where many members are expected near the threshold.

**Search under a budget.** Threshold shifts on a grid are costed on shared draws. A schedule is feasible if the 95th percentile of total cost stays within the budget; feasible schedules are ranked by reach, the expected share of members above the first paid threshold, and then by expected cost.

**Behavioural response.** A whole-spend schedule gives a member just below a threshold a reason to spend the difference: the reward jumps by $\Delta\rho\, t_k$ for an extra spend of at most the gap. In a stress scenario, members within 5 units below a threshold with a jump move up to it with probability 0.5, which is the bunching documented below notches in tax schedules (Kleven, 2016). Marginal schedules have no jumps and are left unchanged.

**Risk.** The 95% value at risk of total cost is the budget exceeded with probability 5%; the expected shortfall is the mean cost when it is exceeded (Acerbi and Tasche, 2002). Monte Carlo standard errors are reported for the mean (sample standard deviation over draws) and for quantiles (batch means). For Gaussian predictive distributions, scrambled Sobol points replace independent draws and lower the variance of the estimated mean (Owen, 2003).

## 10. Effect of the sign-up promotion

Members who joined through the promotion could spend more or less than other members for two reasons: the promotion changed their behaviour, or different people took it up. Only the first is an effect of the promotion.

**Estimands and identification.** With potential outcomes $Y(1)$ and $Y(0)$, the average treatment effect is $\tau_{\text{ATE}} = \mathbb E[Y(1) - Y(0)]$ and the effect on the treated is $\tau_{\text{ATT}} = \mathbb E[Y(1)-Y(0)\mid T=1]$. Both are identified under consistency, unconfoundedness $(Y(0),Y(1)) \perp T \mid X$ and overlap $0 < e(x) = \Pr(T=1\mid X=x) < 1$. The covariates are the attributes fixed at sign-up: region, joining month and tenure band. First-month spend and basket size are measured after joining and may themselves respond to the promotion; adjusting for them would remove part of the effect being estimated, so they are left out.

**Estimators.**

- *Regression adjustment* fits separate linear outcome models $\hat\mu_1, \hat\mu_0$ and averages $\hat\mu_1(x) - \hat\mu_0(x)$ (ATE) or $y - \hat\mu_0(x)$ over the treated (ATT).
- *Inverse probability weighting* with normalised (Hajek) weights $1/\hat e$ and $1/(1-\hat e)$ for the ATE, and $\hat e/(1-\hat e)$ for the controls for the ATT. Propensities are predicted out of fold.
- *Augmented IPW* adds the weighted residuals to the outcome-model contrast,
  $$\psi_i = \hat\mu_1(x_i) - \hat\mu_0(x_i) + \frac{t_i(y_i - \hat\mu_1(x_i))}{\hat e(x_i)} - \frac{(1-t_i)(y_i - \hat\mu_0(x_i))}{1 - \hat e(x_i)},$$
  is consistent if either nuisance model is correct, and with cross-fitted nuisances has standard error $\mathrm{sd}(\psi)/\sqrt n$ (Robins et al., 1994; Chernozhukov et al., 2018).
- *Double machine learning* for $Y = \theta T + g(X) + \varepsilon$ regresses $y - \hat\ell(x)$ on $t - \hat m(x)$ with cross-fitted gradient boosting for $\ell$ and logistic regression for $m$, and takes the median over three sample splits.

**Diagnostics.** Overlap is checked by the range of the propensity in each group and the Kish effective sample size of the weights; the trimming rule of Crump et al. (2009) is available when overlap is poor. Balance is checked by standardized mean differences of every covariate level before and after weighting, with $|d| < 0.1$ as the threshold. A placebo test re-estimates the effect after permuting the promotion labels, within tenure bands if required.

**Sensitivity.** Unconfoundedness cannot be tested. Following Cinelli and Hazlett (2020), the regression estimate is compared with what an omitted confounder $Z$ would do. If $Z$ explains a share $R^2_{T\sim Z\mid X}$ of the residual variance of the promotion and $R^2_{Y\sim Z\mid T,X}$ of the residual variance of spend, the bias is at most

$$\mathrm{se}(\hat\tau)\sqrt{\mathrm{df}}\,\sqrt{\frac{R^2_{Y\sim Z\mid T,X}\,R^2_{T\sim Z\mid X}}{1 - R^2_{T\sim Z\mid X}}}.$$

The robustness value $\mathrm{RV}_q$ is the common value of the two partial $R^2$ that would reduce the estimate by $100q\%$, and bounds for a confounder $k$ times as strong as an observed covariate translate the question into terms of variables the business knows.

**Validation.** `causal/simulate.py` resamples covariates from `train.csv`, generates spend without the promotion from the OLS fit, and assigns the promotion with a logistic propensity that rises with expected spend. With strong confounding the naive difference overstates a true effect of 6 by more than 5, while regression adjustment, AIPW and DML recover it within their intervals; the tests check this for every estimator.

**Who responds.** The conditional effect $\tau(x) = \mathbb E[Y(1)-Y(0)\mid X=x]$ is estimated with five meta-learners (Kunzel et al., 2019): the S-learner (one model with the treatment as a feature), the T-learner (one model per arm), the X-learner (imputed individual effects smoothed per arm and combined with the propensity), the DR-learner (a regression of the AIPW scores on $x$; Kennedy, 2023) and the R-learner (minimising $\sum_i(\tilde y_i - \tau(x_i)\tilde t_i)^2$ on cross-fitted residuals; Nie and Wager, 2021). Their accuracy is measured in simulation by $\mathrm{PEHE} = \{n^{-1}\sum_i(\hat\tau(x_i)-\tau(x_i))^2\}^{1/2}$.

**Inference on heterogeneity.** A CATE model has no standard errors, so its predictions are treated as a proxy $S(x)$ and tested (Chernozhukov et al., 2018b). The best linear predictor $\psi_i = \beta_1 + \beta_2(S(x_i) - \bar S) + \epsilon_i$, with AIPW scores $\psi_i$, gives the ATE as $\beta_1$ and the informativeness of the proxy as $\beta_2$; sorted group average effects (GATES) compare the members with the largest and smallest predicted effects; the classification analysis (CLAN) describes who they are. Each of ten random splits trains the learner and the nuisance models on one half and evaluates on the other; estimates are medians over splits and p-values are twice the median p-value. Nested cross-fitting on the full sample is not used: it lets a unit's score and its neighbours' proxies share training data, and on `train.csv` it produced a significant but spurious heterogeneity loading that changed sign between seeds.

**Targeting.** With margin $m$ and cost $c$ per promotion, the rule $\pi(x) = \mathbb 1\{m\hat\tau(x) > c\}$ is worth $V(\pi) = n^{-1}\sum_i \pi(x_i)(m\psi_i - c)$ per member relative to offering it to nobody, which is unbiased for a rule learned on other data (Athey and Wager, 2021). Under a budget, members are ranked by predicted effect and the top share with a positive predicted net gain is chosen; the value-budget curve shows where extra budget stops paying.

## 10a. Hierarchical Bayesian model

**Model.** $y_i = x_i^\top\beta + u_{g(i)} + \varepsilon_i$ with region effects $u_g \sim \mathcal N(0,\tau^2)$, fixed effects for the amounts, tenure band, joining month and promotion, and inverse-gamma priors $\mathrm{IG}(1,1)$ on $\sigma^2$ and $\tau^2$. Two extensions are available: Student-t errors with $\nu$ degrees of freedom, written as a normal scale mixture $\varepsilon_i \mid \lambda_i \sim \mathcal N(0, \sigma^2/\lambda_i)$, $\lambda_i \sim \mathrm{Gamma}(\nu/2,\nu/2)$ (Geweke, 1993), and a separate residual variance for each tenure band.

**Sampling.** All full conditionals are conjugate. $(\beta, u)$ are drawn jointly from one multivariate normal per sweep: drawing them separately mixes badly, because the intercept and the mean of the region effects are almost unidentified apart. Four chains of 1,500 iterations with 500 discarded are run from dispersed starting values; convergence is checked with split $\hat R$ and the effective sample size of Vehtari et al. (2021).

**Checks and comparison.** Posterior predictive checks compare the mean, standard deviation, extremes, skewness and kurtosis of the observed spend with their distribution over replicated data sets. WAIC (Watanabe, 2010) and the cross-validated log predictive density compare the variants with each other and with OLS. Because there are only four regions, the posterior of $\tau$ is driven by its prior; `prior_sensitivity` refits under vague and informative priors to show which conclusions move.

**Reward cost.** The posterior predictive sampler draws one set of parameters per simulated year for all customers jointly, so parameter uncertainty moves every forecast together, as a wrong model would in practice, and noise is drawn per customer.

## 10b. Explanations and counterfactuals

**Global effects.** Accumulated local effects (Apley and Zhu, 2020) average the change of the forecast across each quantile bin of a numeric attribute over the members actually in that bin, so correlated attributes are never set to combinations that do not occur, as partial dependence does. Centred ICE curves (Goldstein et al., 2015) show each member's forecast as one attribute moves; their spread around the centred partial dependence, relative to its range, measures interaction.

**Stability.** Grouped permutation importance is recomputed on the held-out part of every fold; Kendall's W across folds summarises whether the ranking is a property of the data or of one split.

**Surrogate.** A regression tree of depth 2 to 4 is fitted to the champion's forecasts (not to spend). Its fidelity, the R-squared between tree and champion on held-out members, says how far its printed rules can be trusted as a summary.

**Counterfactuals.** For a member below a reward threshold, bisection finds the smallest first-month spend that lifts the forecast to the next threshold. Region, tenure, joining month and promotion never change, and the search stays below the 99th percentile of first-month spend in the training data. With two actionable attributes, a grid inside their 1st to 99th percentiles is searched for the change with the smallest total distance in units of each attribute's median absolute deviation (Wachter et al., 2018).

## 11. Customer segments

**Distance.** Member attributes are mixed: two amounts, an ordinal tenure band and three nominal attributes. The Gower (1971) distance averages per-attribute dissimilarities in $[0,1]$: the absolute difference divided by the range for amounts, the rank difference divided by the number of levels minus one for tenure (Podani, 1999), and a mismatch indicator for nominal attributes. A missing value removes that attribute from the average for the pairs it is part of.

**Clustering.** K-medoids on the distance matrix, with the alternating algorithm of Park and Jun (2009) and ten restarts from $k$-means++ style seeds; medoids are real members, which makes each segment easy to describe. The number of segments maximises the average silhouette width (Rousseeuw, 1987) among segmentations whose smallest segment holds at least 2% of members.

**Stability.** Each of 30 bootstrap resamples is clustered again, and every original segment is matched with its most similar resampled segment by the Jaccard index on the members in the resample. A mean Jaccard above 0.75 indicates a stable segment and below 0.5 one that dissolves (Hennig, 2007).

**Result on `loyalty.csv`.** Silhouette widths stay between 0.14 and 0.32 for 2 to 8 segments, well below the 0.5 usually read as reasonable structure. The best partition splits the members by region and promotion uptake, and no segment reaches a bootstrap Jaccard of 0.75. The recorded attributes do not contain natural customer groups; segments for marketing would have to come from behavioural data that the extract does not hold.

## 12. Data contracts and drift monitoring

**Contracts.** `configs/contracts.yaml` states, column by column, the kind, the allowed range or levels, uniqueness and a budget for missing values. `loyalty-spend validate` checks a cleaned extract against a contract and lists every violation at once, so a broken extract fails before it reaches a model.

**Univariate drift.** For a feature binned into $B$ bins with reference shares $e_b$ and new shares $a_b$, the population stability index is

$$\mathrm{PSI} = \sum_b (a_b - e_b)\log\frac{a_b}{e_b},$$

the symmetrised Kullback-Leibler divergence of the binned distributions. Numeric features use ten quantile bins of the reference; categorical features use one bin per declared level. Both get extra bins for missing values and for unseen levels, so a rise in missingness or a new region counts as drift. Under no drift, $\mathrm{PSI} \approx \chi^2_{B-1}\,(1/n_1 + 1/n_2)$, so PSI alone overstates drift in small samples: with 250 scoring rows and 13 joining-month levels its expected value is about 0.06. A feature is therefore flagged only when the PSI exceeds the moderate (0.1) or large (0.25) band *and* its two-sample test (Kolmogorov-Smirnov for numeric, chi-square for categorical features) is significant after Benjamini-Hochberg adjustment across features (Benjamini and Hochberg, 1995).

**Joint drift.** A shift in the joint distribution can leave every marginal unchanged, for example if long-tenure members start to come from a different region. A gradient-boosted classifier is trained to separate reference from new rows; its out-of-fold AUC is 0.5 under no shift, and the Mann-Whitney test on the out-of-fold scores gives a p-value. The same classifier provides the density ratios used by weighted conformal prediction.

**Prediction and performance drift.** The PSI and KS test are applied to model outputs, and, once spend is observed for a new cohort, to the target. The change in RMSE is reported as a ratio with a bootstrap percentile interval.

## 13. Assumptions and limitations

- The data are a small synthetic sample from a certification exercise. Results on `train.csv` describe that sample; the simulation study exists because those data cannot discriminate between methods.
- All validation is cross-sectional. There is no time dimension, so drift between programme years is not tested.
- The reward schedule in `configs/default.yaml` is an illustrative assumption; the brief does not state the real one.
- Conformal guarantees are marginal and assume exchangeability between calibration and scoring customers. Under a change in the customer mix that leaves the relation between features and spend intact, weighted conformal prediction restores coverage only as far as the density ratio is estimated well; a change in that relation itself is not covered by any of the methods.
- Shapley decompositions of $R^2$ describe a linear model's explained variance, not causal effects. Tenure is associated with spend; whether extending a customer's tenure would raise spend is a different, causal question.

## References

- Acerbi, C., and Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking and Finance*, 26(7), 1487-1503.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623-2631.
- Apley, D. W., and Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. *Journal of the Royal Statistical Society, Series B*, 82(4), 1059-1086.
- Athey, S., and Wager, S. (2021). Policy learning with observational data. *Econometrica*, 89(1), 133-161.
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. *Statistics in Medicine*, 28(25), 3083-3107.
- Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486-507.
- Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, 57(1), 289-300.
- Cawley, G. C., and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079-2107.
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.
- Chernozhukov, V., Demirer, M., Duflo, E., and Fernandez-Val, I. (2018b). Generic machine learning inference on heterogeneous treatment effects in randomized experiments. *NBER Working Paper* 24678.
- Chernozhukov, V., Fernandez-Val, I., and Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica*, 78(3), 1093-1125.
- Cinelli, C., and Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society, Series B*, 82(1), 39-67.
- Crump, R. K., Hotz, V. J., Imbens, G. W., and Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
- Dawid, A. P. (1984). Statistical theory: The prequential approach. *Journal of the Royal Statistical Society, Series A*, 147(2), 278-292.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Geweke, J. (1993). Bayesian treatment of the independent Student-t linear model. *Journal of Applied Econometrics*, 8(S1), S19-S40.
- Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.
- Gneiting, T., Raftery, A. E., Westveld, A. H., and Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *Monthly Weather Review*, 133(5), 1098-1118.
- Goldstein, A., Kapelner, A., Bleich, J., and Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. *Journal of Computational and Graphical Statistics*, 24(1), 44-65.
- Gorishniy, Y., Rubachev, I., and Babenko, A. (2022). On embeddings for numerical features in tabular deep learning. *Advances in Neural Information Processing Systems*, 35.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems*, 34.
- Gower, J. C. (1971). A general coefficient of similarity and some of its properties. *Biometrics*, 27(4), 857-871.
- Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems*, 35, Datasets and Benchmarks Track.
- Groemping, U. (2006). Relative importance for linear regression in R: The package relaimpo. *Journal of Statistical Software*, 17(1), 1-27.
- Guo, C., and Berkhahn, F. (2016). Entity embeddings of categorical variables. *arXiv:1604.06737*.
- Hennig, C. (2007). Cluster-wise assessment of cluster stability. *Computational Statistics and Data Analysis*, 52(1), 258-271.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Kendall, M. G., and Babington Smith, B. (1939). The problem of m rankings. *Annals of Mathematical Statistics*, 10(3), 275-287.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*, 17(2), 3008-3049.
- Kleven, H. J. (2016). Bunching. *Annual Review of Economics*, 8, 435-464.
- Koenker, R., and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156-4165.
- Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*, 30.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094-1111.
- Lindeman, R. H., Merenda, P. F., and Gold, R. Z. (1980). *Introduction to Bivariate and Multivariate Analysis*. Scott, Foresman.
- Nadeau, C., and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
- Nie, X., and Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
- Owen, A. B. (2003). Quasi-Monte Carlo sampling. In *Monte Carlo Ray Tracing: SIGGRAPH 2003 Course 44*, 69-88.
- Papadopoulos, H., Gammerman, A., and Vovk, V. (2008). Normalized nonconformity measures for regression conformal prediction. *Proceedings of the IASTED International Conference on Artificial Intelligence and Applications*, 64-69.
- Park, H.-S., and Jun, C.-H. (2009). A simple and fast algorithm for K-medoids clustering. *Expert Systems with Applications*, 36(2), 3336-3341.
- Podani, J. (1999). Extending Gower's general coefficient of similarity to ordinal characters. *Taxon*, 48(2), 331-340.
- Robins, J. M., Rotnitzky, A., and Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89(427), 846-866.
- Romano, Y., Patterson, E., and Candes, E. J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
- Tibshirani, R. J., Barber, R. F., Candes, E. J., and Ramdas, A. (2019). Conformal prediction under covariate shift. *Advances in Neural Information Processing Systems*, 32.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Burkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
- Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Vovk, V., Shen, J., Manokhin, V., and Xie, M. (2019). Nonparametric predictive distributions based on conformal prediction. *Machine Learning*, 108(3), 445-474.
- Wachter, S., Mittelstadt, B., and Russell, C. (2018). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. *Harvard Journal of Law and Technology*, 31(2), 841-887.
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. *Journal of Machine Learning Research*, 11, 3571-3594.
- Welch, B. L. (1951). On the comparison of several mean values: An alternative approach. *Biometrika*, 38(3/4), 330-336.
