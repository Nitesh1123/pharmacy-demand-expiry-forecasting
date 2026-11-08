# Held-out test evaluation — November 8, 2026 (Day 3)

Planned for Day 6 and completed early. The model selections were locked on validation,
and the evaluation code was committed before the test set was scored. The test set
was scored once, and nothing was changed because of these results. Results are from
synthetic stress-test data and are not evidence of real-pharmacy accuracy.

Models are the ones validated, fitted on training rows only. Machine-readable results
are in [test-evaluation.json](test-evaluation.json).

## Demand: 1,040 test rows (13 Sunday origins × 80 products)

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| **Tuned gradient boosting (served)** | **26.74** | 36.01 | **4.39%** |
| Seasonal naive | 27.51 | **35.56** | 4.52% |
| 28-day moving average × 30 | 43.55 | 78.61 | 7.15% |
| Ridge | 49.38 | 74.64 | 8.11% |

Gradient boosting is slightly ahead of seasonal naive on MAE and WAPE and slightly
behind on RMSE, so the two are close. It clearly beats the moving average, which is
the baseline that doesn't benefit from the simulator's exact yearly repeat. The mean
of actual minus predicted is 0.38 units, so there is no meaningful overall bias.

- By origin date, the served model's WAPE ranges from 3.87% to 4.79%.
- By category, it ranges from 3.65% (analgesics) to 5.71% (antibiotics).
- It beats seasonal naive's WAPE in four of six categories. Antipyretics and
  respiratory, the two strongly seasonal categories, go to seasonal naive.

### Influential inputs

Permutation importance on test rows, as the rise in MAE when one input is shuffled:

| Input | MAE increase |
|---|---:|
| `sales_14d` | 82.3 |
| `sales_56d` | 70.0 |
| `sales_28d` | 63.6 |
| `sales_7d` | 54.3 |
| `sales_same_window_last_year` | 29.8 |
| `week_of_year` | 10.4 |
| `category` | 7.5 |

Rolling sales inputs are strongly correlated, so shuffling one understates the group.
`mean_28d` shows 0 because it duplicates `sales_28d`, not because recent sales don't
matter. These measure what the model relies on, not what causes demand.

### One prediction and error explained

The largest error was product MED-051 (antipyretics), origin 2023-11-19. It sold 1,610
units over the next 30 days. The model predicted 1,425, about 12% too low; seasonal
naive predicted 1,593.

Setting one input at a time to its typical training value shows what drove the forecast
up: recent 14-day sales of 561 (+267 units), 7-day sales of 275 (+255), 56-day sales of
2,309 (+146) and the late-November week (+116). `unit_price` also shows +91, but price
doesn't drive demand in the simulator, so that effect is association, not cause.

The target window runs November 20 to December 19, and antipyretic demand steps up 40%
in December. The model has seen only one earlier November-to-December transition, so it
under-forecast the step. Seasonal naive copied last year's matching window.

## Expiry alerts: 103 test batches (71 risky, 32 not)

| Model | Caught | Missed | False alarms | Correct non-alerts | Precision | Recall | F1 (95% interval) |
|---|---:|---:|---:|---:|---:|---:|---|
| **Rule (served)** | 67 | 4 | 5 | 27 | 0.93 | 0.94 | **0.94** (0.89–0.97) |
| Gradient boosting (cutoff 0.55) | 67 | 4 | 3 | 29 | 0.96 | 0.94 | 0.95 (0.91–0.98) |
| Logistic regression (cutoff 0.55) | 66 | 5 | 5 | 27 | 0.93 | 0.93 | 0.93 (0.88–0.97) |

Gradient boosting has two fewer false alarms than the rule on this test set. The
selection stays with the rule: the test set can't be used to change a locked choice,
the intervals overlap heavily, and validation didn't show a clear advantage.

### Why the rule was wrong

All nine errors were traced through the sales records after the evaluation.

- **False alarms (5 of 5): stock counted as ahead in the queue expired before demand
  reached it.** For example, BAT-01765 had 1,196 units ahead, so the rule projected no
  sales for it. Only 138 of those units sold before they expired, and demand moved on
  to this batch, which sold out all 833 units.
- **Missed, later delivery jumped the queue (2 of 4).** BAT-01803 had nothing ahead of
  it at the snapshot. A later delivery with an earlier expiry then took 1,021 units of
  demand, and this batch sold only 297 and expired 54% unsold. BAT-01779 is the same
  pattern, with 635 units. A forecast can't see deliveries that haven't happened yet.
- **Missed, demand fell (1 of 4).** BAT-01712 used a 45.5 per day rate that included
  August's monsoon peak; actual demand was 34.6 per day.
- **Missed, close to the line (1 of 4).** BAT-01762 was projected at 14.8% leftover and
  ended at 21.1%, just over the 20% label.

Possible improvement, not applied: count only queued units that can sell before their
own expiry. This idea came from test-set errors, so testing it would need new data.

## Limitations

- Synthetic stress-test data: short-dated and oversized deliveries are deliberately
  common, so alert rates aren't realistic.
- There are no real stockouts, lead times, returns or promotions.
- The test period covers one autumn, with targets ending in December. Other seasons
  aren't tested.
- Seasonal naive benefits from the simulator repeating the same seasonality each year.
- 103 batches is a modest sample, so intervals are wide enough that small differences
  between expiry models aren't meaningful.
- Outputs are advisory stock guidance for human review, never medication or treatment
  advice.

## Verification

- 62 tests pass. The evaluation tests use a separate seed-7 dataset, so they never
  touched the default test split.
- Tests confirm that changing validation or test labels leaves fitted models unchanged.
  A one-off mutation check confirmed that test fails if held-out rows are fitted.
- CI runs this evaluation but doesn't compare the report byte for byte, because boosted
  model floats can differ slightly between operating systems.
