# Day 2 verification — November 7, 2026

Scope: data generation, data-quality checks, reproducibility, and a corrected
plan for the November 15 submission. No models have been trained yet.

## Changes and evidence

The original seeded generator produced only eight expired batches with more than
20% leftover stock. It also exposed only final balances, which are unsuitable for
historical predictors. Version 2 adds documented short-dated/oversized deliveries,
receipt dates, Sunday end-of-day snapshots, complete product-day sales, and
observed outcome labels. Unfinished outcomes remain unknown.

Default run: seed 42, 80 products, 730 days, 2022-01-01 through 2023-12-31.

| Output | Rows |
|---|---:|
| Products | 80 |
| Batches | 2,003 |
| Batch sales allocations | 59,281 |
| Daily product sales | 58,400 |
| Weekly active batch snapshots | 39,062 |

Observed batch outcomes: **389 positive, 1,063 negative, 551 unknown**.
The run includes 31 zero-sales product-days and zero stockout product-days.
See [machine-readable validation](day2-data-quality.json) and
[file hashes/configuration](../data/manifest.json).

The PRD's proposed first-snapshot-within-60-days selection was checked separately
for data support. After restricting to labels available before each boundary:

| Proposed period | Independent positive batches | Independent negative batches |
|---|---:|---:|
| Train | 262 | 179 |
| Validation | 25 | 29 |
| Test | 71 | 32 |

These are feasibility counts, not metrics. The split implementation and leakage
tests belong to Day 3. Validation still has limited support: report uncertainty
and avoid broad hyperparameter searches.

## Verification performed

- Created a fresh CPython 3.13 virtual environment and installed the pinned stack.
- All 22 tests passed: seeded repeatability, unchanged historical prefixes,
  observed/unknown outcomes, historical stock reconstruction, FEFO order, complete
  daily coverage, invalid CLI inputs, and rejection of corrupt datasets.
- Ran the default generator and validator end to end; stock balances, temporal
  validity, snapshot coverage, foreign keys and labels passed.
- `pip check` found no dependency conflicts.
- Added Windows/Linux CI to run tests, regenerate data, validate it, and compare
  CSV hashes and the quality report against the checked-in default results.

## Limitations and next work

Synthetic expiry prevalence is deliberately increased for experimentation and is
not calibrated to real pharmacies. Immediate replenishment removes stockouts in
the default run; no returns, lead times, promotions or multi-store effects exist.
Final batch balances and simulator-only demand fields are excluded from features.

Day 3: EDA notebook, feature pipeline, temporal splits with target-window gaps,
and tests proving features use only information available at prediction time.
Training, inference, explanation, final report and video remain scheduled in the PRD.
