# Pharmacy demand and expiry-risk project requirements

Updated November 8, 2026 (Day 3). Source of acceptance criteria: the assessment
document in the repository root. Submission date: November 15, 2026; use the
invitation for the exact cutoff and timezone. Target completion: November 14.

## Scope and current status

Build a reproducible, explainable single-store demonstration that forecasts
medicine demand and flags batches likely to expire with material leftover stock.
Outputs are advisory stock rotation/replenishment guidance, never treatment advice.

Day 2 implemented synthetic generation, observed batch outcomes, weekly inventory
snapshots, data-quality checks, tests, reproducible installation, and CI. Day 3 added
the EDA notebook, point-in-time feature tables (`src/features.py`), the split rules
below (`src/splits.py`), and truncation-based leakage tests. Model training,
evaluation, artifacts, and the app remain planned.

## Prediction contract

Demand: at the end of date `t`, predict total recorded units sold per product
over `t+1` through `t+30` inclusive. This resolves the earlier daily/weekly/horizon
ambiguity. A daily trajectory is optional; every required model comparison uses
the same 30-day total.

Expiry: at a Sunday end-of-day snapshot, predict whether a currently stocked
batch will expire with **more than 20% of its original received quantity unsold**.
This is a binary observed outcome; an alert is a prediction of that outcome.
Risk probabilities and an alert threshold will be calibrated/selected on
validation data, not the final test set. A transparent rule is the baseline.

Unexpired batches with stock at the end of observed history have unknown
outcomes. Sold-out batches are resolved negatives under the no-returns assumption.
The current implementation records these distinctions; it does not train a model.

## Data and leakage controls

Use the [versioned data contract](../data/README.md). Historical stock comes from
dated snapshots or reconstruction through the prediction date. Never use final
balances, future outcomes, or simulator-only demand parameters as predictors.
All imputation, encoding and other learned transforms fit on training data only.

Planned temporal boundaries for the default history:

| Set | Prediction origins | Additional eligibility |
|---|---|---|
| Train | Before 2023-06-01, after feature warm-up | Every target outcome observed before 2023-06-01 |
| Validation | 2023-06-01 through 2023-08-31 | Every target outcome observed before 2023-09-01 |
| Test | From 2023-09-01 | Outcomes fully observed by 2023-12-31 |

For demand, the target ends at `t+30`; remove origins whose windows cross a
boundary. For batch risk, require `outcome_date` before the relevant boundary.
Do not split a stacked product table by row index. All products share date cutoffs.

To avoid repeated batches straddling splits, begin expiry evaluation with one
eligible weekly snapshot per batch: the first snapshot within 60 days of expiry.
Count positives and negatives after this selection and outcome-availability
filtering, and report exclusions. This focuses the demonstration on near-expiry
stock; it does not establish performance for batches years from expiry. If data
support is insufficient, revise the simulation before model development rather
than fabricating labels or repeatedly tuning against final test outcomes.

## Modeling and acceptance

1. EDA covers data quality, class support, demand patterns, zero days, category
   differences, stockout limitations, and leakage analysis. Statistical tests are
   diagnostics, not automatic rules that decide which model is valid.
2. Compare a moving-average/seasonal baseline and two reasonable candidates
   where feasible. Start with a simple linear model and a tree-based model;
   justify the choice using validation evidence. Prophet/deep learning is optional.
3. Use identical tasks and evaluation origins, with model-appropriate preprocessing.
   Report MAE, RMSE, WAPE for demand; precision, recall, F1 and confusion matrix
   for batch alerts. Define zero-denominator behavior. Choose the alert threshold
   on validation data and report independent batch support and uncertainty.
4. Run final test evaluation after model selection. Explain a prediction and an
   error; state that synthetic performance does not demonstrate real-world accuracy.
5. One Streamlit app loading a saved artifact is sufficient. FastAPI is optional.
6. A fresh clone must reproduce data, splits, training, evaluation and inference.
   Include preprocessing/inference tests as those components are implemented.

## Delivery plan

| Date (2026) | Deliverable |
|---|---|
| Nov 7, Saturday | Correct data/outcomes, quality tests, reproducible setup, Day 2 PR |
| Nov 8, Sunday | EDA, point-in-time features, temporal split implementation |
| Nov 9, Monday | Baselines and first candidate models |
| Nov 10, Tuesday | Candidate comparison and validation tuning |
| Nov 11, Wednesday | Final evaluation, error analysis, explanation |
| Nov 12, Thursday | Saved artifact and working inference app |
| Nov 13, Friday | Reproduction and failure-case tests |
| Nov 14, Saturday | Report, README, 3–5 minute video, accessible links |
| Nov 15, Sunday | Verification and submission before the invitation cutoff |

## Submission evidence

Repository/commit, generator and data provenance, technical report with EDA and
metrics, reproducible artifact generation, inference demonstration, and a 3–5
minute video. Local-only demos are allowed if declared. Verify reviewer access.
Complete the form's tool-use/source disclosures accurately; Git authorship remains
the candidate's and is separate from the assignment's disclosure requirements.
