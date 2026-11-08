# Day 3 model comparison — November 8, 2026

Scope: baselines and first candidate models, compared on the **validation split
only**. The test split has not been read by any training or scoring code. No model
artifact is saved yet. The two tables below use fixed defaults and a 0.5 probability
cutoff; narrow tuning and the final selection follow in their own section.

## Delivered

- `src/metrics.py`: MAE, RMSE, WAPE, precision, recall, F1, confusion counts and
  bootstrap 95% intervals over independent batches. WAPE, precision, recall and F1
  are reported as undefined (`null`), not zero, when their denominator is zero.
- `src/train.py`: fits on training rows only and scores validation rows only.
- [Machine-readable results](day3-validation.json).

## Demand: 720 validation rows (9 Sunday origins × 80 products)

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal naive (same 30 days last year) | **28.92** | **38.62** | **4.51%** |
| Gradient boosting | 30.61 | 44.07 | 4.77% |
| 28-day moving average × 30 | 49.56 | 81.74 | 7.72% |
| Ridge | 55.15 | 80.93 | 8.59% |

Gradient boosting clearly beats the moving average, but neither candidate beats
seasonal naive.

- The simulator repeats the same calendar seasonality every year, which favours
  seasonal naive. Real demand is unlikely to repeat this cleanly.
- Validation origins run only from June to July, so they cover one seasonal
  transition, the rise into the monsoon months. A moving average lags that rise.
- Ridge has a lower RMSE than the moving average but a higher MAE. One likely cause,
  not yet tested, is that the last-year feature is missing for 71% of training rows.

## Expiry: 54 validation batches (25 positive, 29 negative)

| Model | TP | FP | FN | TN | Precision | Recall | F1 (95% interval) |
|---|---:|---:|---:|---:|---:|---:|---|
| Rule: projected leftover > 20% | 24 | 2 | 1 | 27 | 0.92 | 0.96 | **0.94** (0.86–1.00) |
| Gradient boosting | 24 | 3 | 1 | 26 | 0.89 | 0.96 | 0.92 (0.83–0.99) |
| Logistic regression | 23 | 2 | 2 | 27 | 0.92 | 0.92 | 0.92 (0.83–0.98) |

The three models differ by one or two batches and their intervals overlap heavily,
so this validation set cannot separate them. The candidates have not shown an
advantage over the transparent rule.

## Narrow tuning and selection

Hyperparameters and cutoffs were chosen on the same validation rows they are scored
on, so these selected scores are optimistic. The test set is still unopened.

**Demand.** Nine gradient-boosting settings were fitted on training rows and compared
on validation MAE. The best used 400 iterations and 31 leaves: MAE 28.71, RMSE 40.88,
WAPE 4.47%. Seasonal naive remains at MAE 28.92, RMSE 38.62, WAPE 4.51%.

- Tuned boosting is nominally ahead on MAE and WAPE, and behind on RMSE. After tuning
  on these rows, that is not evidence that it is better.
- The best setting sits at the largest values in the grid. The grid was not widened,
  because nine validation origins cannot support a broader search.

**Expiry.** Each candidate's probability cutoff was chosen for the highest validation
F1, on a 0.10 to 0.90 grid. The rule stays selected.

| Model | Cutoff | TP | FP | FN | TN | F1 | F1 minus rule (95% paired interval) |
|---|---:|---:|---:|---:|---:|---:|---|
| Rule | n/a | 24 | 2 | 1 | 27 | 0.941 | n/a |
| Gradient boosting | 0.55 | 24 | 2 | 1 | 27 | 0.941 | 0.000 to 0.000 |
| Logistic regression | 0.55 | 23 | 1 | 2 | 28 | 0.939 | -0.062 to 0.055 |

Neither candidate's interval is entirely above zero, even with a cutoff tuned on these
rows, so the rule stays. At its cutoff, gradient boosting makes the same decision as the
rule on all 54 batches, so its interval is exactly zero.

## Leakage and evaluation evidence

- Changing validation labels leaves every fitted model's predictions unchanged.
- Removing test rows leaves both validation reports identical.
- Models receive exactly the declared feature lists. Boosting early stopping is off
  because it would draw a random, non-chronological holdout.
- One-off mutation check: fitting on validation rows, or scoring test rows, made the
  corresponding test fail.

## Verification

- 55 tests pass and `pip check` is clean.
- `python -m src.train --report reports/day3-validation.json` runs end to end.
- CI runs training but does not compare this report byte for byte, because boosted
  model floats can differ slightly between operating systems.

## Limits

- Results are from synthetic stress-test data and say nothing about real pharmacies.
- Tuning was deliberately narrow, and selected scores are optimistic.

## Next (Day 4, November 9)

- Expiry decision: the transparent rule stays the alert method unless a candidate
  clearly beats it on validation, meaning the 95% paired bootstrap interval for the
  F1 difference is entirely above zero. On today's results no candidate has an F1
  above the rule's.
- Demand decision: serve gradient boosting as the trained artifact. Seasonal naive
  had slightly lower validation error, and the final report keeps it as the reference
  and shows both on the test set.
- The single test-set evaluation has since been run with the selections above; see
  the [held-out test evaluation](test-evaluation.md).
