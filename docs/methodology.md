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
