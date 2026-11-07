# Synthetic data contract (schema version 2)

All data is generated locally. No patient, personal, confidential or licensed
third-party dataset is used. Reproduce it with `python src/data_generator.py`.
The fixed default configuration and dependency versions are recorded in
`manifest.json`; CSV hashes let reviewers check reproducibility.

## Simulation assumptions

- One store, 80 synthetic products, daily history from 2022-01-01 to 2023-12-31.
- Six categories, Poisson demand with category seasonality and weekend effects.
- Reorder when usable stock falls below ten days of baseline demand; an
  independent 3% daily chance can trigger an additional delivery.
- Each delivery has a 30% chance of arriving with 7–60 days remaining shelf
  life; otherwise 60–100% of the product's typical shelf life remains.
- An independent 15% chance triples a delivery's usual 25–59 days of baseline
  demand. These are **stress-test assumptions**, not measured pharmacy rates.
- Manufactured date is expiry minus typical shelf life. Receipt date is explicit
  and can be much later than manufacture.
- Allocation is FEFO (first expiry, first out), with insertion order breaking
  ties. No sales occur on or after the expiry date.
- No returns, transfers, lead times, promotions, or lost records. Restocking is
  immediate, so the default run has no stockouts. This limits realism.

## Tables

| File | Grain and key | Main fields |
|---|---|---|
| `products.csv` | One row per `product_id` | name, category, price, typical shelf life, simulator baseline demand |
| `batches.csv` | One row per `batch_id` | product, manufacture/receipt/expiry dates, received quantity, final remaining quantity, outcome status/date, nullable expiry label |
| `sales.csv` | One positive batch allocation per `sale_id` | date, product, batch, quantity, price; a product-day can span batches |
| `daily_sales.csv` | Every `(date, product_id)`, including zero sales | quantity sold, simulated demand, unmet demand |
| `inventory_snapshots.csv` | Each active `(as_of_date, batch_id)` on Sunday after sales | product and remaining quantity at that time |

An active batch has positive stock and expires after the snapshot date. Sold-out
and expired batches do not appear in subsequent snapshots. A simulation shorter
than the first Sunday still writes a snapshot CSV with headers.

`batches.quantity_remaining` is the final unconsumed balance, including units
that expired. It is **not current saleable shelf stock** and must not be used as
a historical feature. Use `inventory_snapshots.quantity_remaining` instead.

## Observed outcomes

- `sold_out`: depleted before expiry; label `0`, outcome date is the final sale.
  This is a known negative because the simulation has no returns.
- `expired`: expired with remaining units; label `1` only when remaining units
  are **strictly greater than 20%** of received units, otherwise `0`.
- `censored`: still unexpired with remaining stock at history end; label and
  outcome date are blank. Its eventual outcome is unknown, not safe.

The 20% cutoff is a declared demonstration business rule, not a clinical threshold.
Labels come from observed simulation outcomes, never predictions.

## Fields excluded from model inputs

- `products.base_daily_demand` is a simulator-only latent parameter.
- `daily_sales.simulated_demand` and `unmet_demand` are oracle diagnostics,
  not fields expected from real transaction data.
- Final batch balance, `outcome_status`, `outcome_date`, and `expiry_risk` are
  evaluation/label fields. They cannot be predictor columns.
- Future sales or future batch deliveries cannot be included at an earlier
  prediction date. Metadata joins must respect `received_date`.

Future demand targets use recorded `quantity_sold`; discuss stockout censoring
if unmet demand appears in another configuration. Features use only data through
the end-of-day prediction origin. Weekly snapshots of one batch are correlated;
evaluation must report independent batch counts, not only snapshot-row counts.

## Derived feature tables

`python -m src.features` writes two ignored CSVs and a split summary
(`reports/day3-splits.json`):

- `demand_features.csv`: one row per Sunday origin and product after a 56-day
  warm-up. Rolling sales (7/14/28/56 days, ending on the origin), same 30-day
  window one year earlier, snapshot stock on hand, calendar and product metadata.
  Target `target_30d` is recorded units over origin+1 to origin+30.
- `expiry_features.csv`: one row per batch at its first Sunday snapshot within
  60 days of expiry. Snapshot stock, FEFO units ahead of the batch, recent product
  sales, and a heuristic projected leftover fraction. Labels are copied from
  `batches.csv` and are never features.

Each row carries a `split` from shared calendar boundaries; rows whose target or
outcome is unknown, or resolves across a boundary, are `excluded`.

## Validation

`python -m src.data_quality` fails on invalid keys, missing values, foreign keys,
negative/fractional quantities, invalid dates, stock imbalance, incomplete daily
coverage, incorrect historical balances, missing weekly snapshots, or invalid
outcome labels. It reports class counts by **outcome date**, which are diagnostics
and must not be mistaken for leakage-safe model splits.
