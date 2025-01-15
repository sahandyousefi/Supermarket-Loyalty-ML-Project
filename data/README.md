# Data

Three files are supplied with the original brief. They are small and are kept in the repository so that every result can be reproduced without a download step.

| File | Rows | Use |
| --- | --- | --- |
| `raw/loyalty.csv` | 1,246 | Raw extract with quality issues; used for cleaning (Task 1) and the driver analysis (Task 2) |
| `raw/train.csv` | 996 | Clean training data with the target; used for all models |
| `raw/test.csv` | 250 | Customers to score; no target |

## Data dictionary

| Column | Type | Rule |
| --- | --- | --- |
| `customer_id` | identifier | Unique, never missing |
| `spend` | continuous | Total spend in the last full year, positive, two decimals; missing replaced by 0 |
| `first_month` | continuous | Spend in the first month of the year, positive, two decimals; missing replaced by 0 |
| `items_in_first_month` | discrete | Items bought in the first month, integer at least 0; missing replaced by 0 |
| `region` | nominal | Americas, Asia/Pacific, Europe, Middle East/Africa; missing replaced by "Unknown" |
| `loyalty_years` | ordinal | 0-1, 1-3, 3-5, 5-10, 10+; missing replaced by "0-1" |
| `joining_month` | nominal | Jan to Dec; missing replaced by "Unknown" |
| `promotion` | nominal | Yes or No; missing replaced by "No" |

## Known issues in `loyalty.csv`

- 125 values of `first_month` are the placeholder `.`.
- 125 values of `joining_month` are empty.
- 125 values of `promotion` are upper case (`YES`, `NO`).

`loyalty-spend clean` applies the dictionary rules and prints a log of every change. The modeling code reads the files with `impute=False`, so a missing `first_month` stays missing and is handled by median imputation with a missing-value indicator instead of being set to zero.

## Relationship between the files

`train.csv` and `test.csv` together contain the same 1,246 customer ids as `loyalty.csv`, but their attributes do not match: region agrees for 25% of shared ids, which is the agreement expected by chance (Cohen's kappa about 0). The files are therefore treated as independent samples and are never joined. `loyalty-spend report` includes this check.
