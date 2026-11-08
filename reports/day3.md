# Day 3 verification — November 8, 2026

Scope: EDA, point-in-time features, chronological splits and leakage tests. No
model has been trained and no validation or test outcome was used to choose
features.

## Delivered

- `src/splits.py`: one set of calendar boundaries for every product. Validation
  starts 2023-06-01 and test starts 2023-09-01. A demand origin belongs to a period
  only if its whole t+1..t+30 target is observed inside that period. A batch belongs
  only if its observed outcome is known before the next boundary. Unknown outcomes
  are always excluded, never treated as negatives.
- `src/features.py` writes two tables (see the [data contract](../data/README.md)):
  - Demand: 7,360 rows (Sunday origin × product).
  - Expiry: 686 rows (one near-expiry snapshot per batch).

  Inference mode keeps recent origins that do not yet have a target.
- [EDA notebook](../notebooks/01_eda.ipynb), executed with outputs, covering data
  quality, outliers, stockout limits, seasonality, weekday effects, target and
  feature relationships (training data only), label support and leakage.

## Split support ([machine-readable](day3-splits.json))

| Period | Demand origins × products | Expiry batches (positive / negative) |
|---|---:|---:|
| Train | 62 × 80 = 4,960 | 236 / 157 |
| Validation | 9 × 80 = 720 | 25 / 29 |
| Test | 13 × 80 = 1,040 | 71 / 32 |
| Excluded | 640 (targets cross a boundary) | 136 (32 unknown, 56 resolve after a boundary, 48 warm-up) |

Validation and test match the Day 2 feasibility analysis exactly. Train is Day 2's
262 / 179 minus 48 batches seen in the first 27 days, which have no 28-day sales rate
and fall inside the PRD's feature warm-up. A test locks these counts in. They describe
data support, not model performance.

## Leakage evidence

- **Truncation test.** The generator is prefix-stable, so a 114-day run is a 240-day
  run with the future removed. Features computed on both runs match for every
  origin and snapshot up to the last day of the short run.
- **Mutation check.** Shifting `sales_7d` one or three days into the future, or
  replacing snapshot stock with the final batch balance, makes the leakage tests
  fail.
- **Forbidden fields.** Feature lists cannot contain simulator-only fields,
  outcomes or targets. A test enforces this.
- **Bug found and fixed.** An origin whose 30-day window ran past the end of the
  data was labelled `train`. The split rule now requires the target or outcome to
  be fully observed. A regression test covers this.
- **Bug found and fixed.** Expiry rows without a full 28-day sales rate were kept in
  training, although the PRD starts training after feature warm-up. The rule baseline
  also read them as "no alert". They are now excluded, and a test covers this.

## EDA findings that shape modeling

- The data passes every integrity check. The only missing values are unknown
  outcomes, which stay unknown. Same-window-last-year sales are undefined in the
  first year, which is 71% of training demand rows, so that feature needs a model
  that handles missing values or imputation fitted on training data.
- Recent sales nearly determine the 30-day total: Spearman correlation is 0.99 on
  training data. The 28-day moving average scaled to 30 days is therefore a strong
  baseline.
- Respiratory and antipyretic products peak in winter and the monsoon, and vitamins
  peak in Jan–Feb. Weekends run about 20% above weekdays; this mostly averages out
  over a 30-day window.
- Expiry baseline rule: alert when projected leftover is above 20% of received
  units. On training data it gives 218 true positives, 18 false negatives, 11 false
  positives and 146 true negatives.
- Many positive batches are short-dated deliveries first seen fewer than 30 days
  before expiry. This separation partly reflects the stress scenario.
- Validation support is small: 54 batches. Report bootstrap intervals and keep
  tuning narrow.

## Verification

- 36 tests pass. The suite includes the leakage, split-boundary and feasibility
  tests.
- The default generator, the validator, `python -m src.features` and the notebook
  all run end to end. CI now also builds the features, executes the notebook, and
  compares `reports/day3-splits.json` with the checked-in copy.
- Dependencies are pinned in `requirements.txt` for Python 3.13, now including
  matplotlib and the notebook runner.

## Next

The baselines and candidates below were finished the same day; see the
[Day 3 model comparison](day3-models.md).

- Demand: moving-average and seasonal-naive baselines, a regularised linear model
  and a gradient-boosted tree.
- Expiry: the rule baseline, logistic regression and a tree model.
- Fit all transforms on training data only, select on validation, and leave the
  test set untouched until final evaluation.
