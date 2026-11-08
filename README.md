# Pharmacy Medicine Demand & Expiry-Risk Forecasting

An individual ML project for pharmacy inventory decision support. The intended
outputs are a 30-day product demand forecast and batch expiry-risk alerts.
Outputs are advisory inventory guidance, never medication or treatment advice.

## Current status: Day 3 — November 8, 2026

Implemented: reproducible synthetic data, weekly inventory snapshots, observed
expiry outcomes, validation, point-in-time feature tables, chronological
train/validation/test splits, an executed EDA notebook, leakage tests, and
Windows/Linux CI. Model training, evaluation, model artifacts, and the inference
app are **not implemented yet**.

See the [PRD and schedule](docs/PRD.md), [data contract](data/README.md),
[EDA notebook](notebooks/01_eda.ipynb), and the [Day 2](reports/day2.md) and
[Day 3](reports/day3.md) reports. Submission is November 15, 2026;
the individualized invitation specifies the exact time and timezone.

## Setup

Use **Python 3.13**. The dependency file pins the implemented data, testing and notebook stack,
including transitive dependencies. Additional model and app packages will be
added and tested as those features are implemented.

Windows PowerShell (activation is not required):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/data_generator.py
.\.venv\Scripts\python.exe -m src.data_quality --report reports/day2-data-quality.json
.\.venv\Scripts\python.exe -m src.features --report reports/day3-splits.json
.\.venv\Scripts\jupyter.exe execute --inplace notebooks/01_eda.ipynb
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Linux/macOS with Python 3.13 installed:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/data_generator.py
.venv/bin/python -m src.data_quality --report reports/day2-data-quality.json
.venv/bin/python -m src.features --report reports/day3-splits.json
.venv/bin/jupyter execute --inplace notebooks/01_eda.ipynb
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

The generator defaults to seed 42, 80 products, and 730 days starting on
2022-01-01. It writes five CSVs and a manifest with versions and SHA-256 hashes
to `data/`. CSVs are ignored by Git; the default manifest and quality report
are checked in so reviewers can compare their own generated data.

For a separate experiment, use another directory:

```powershell
.\.venv\Scripts\python.exe src/data_generator.py --seed 7 --products 10 --days 90 --output_dir data/experiment
.\.venv\Scripts\python.exe -m src.data_quality --data-dir data/experiment
```

Generator configuration comes from CLI arguments. `.env.example` is reserved
for the later app; the current pipeline does not load `.env`.

## Data flow

```text
data_generator.py -> products / batches / sales / daily_sales / inventory_snapshots
                  -> manifest.json (configuration, versions, file hashes)
data_quality.py   -> relational, temporal, balance and outcome checks -> report
features.py       -> demand_features.csv (Sunday origin x product, 30-day target)
                  -> expiry_features.csv (one near-expiry snapshot per batch)
splits.py         -> shared calendar boundaries; targets/outcomes never cross them
notebooks/01_eda  -> quality, demand patterns, label support, leakage analysis

Next: baseline/candidates -> validation selection -> test evaluation -> saved model -> app
```

## Evaluation plan

- Demand: moving-average or seasonal-naive baseline plus justified candidates;
  MAE, RMSE, WAPE on complete future 30-day windows.
- Expiry: compare predicted alerts with actual leftovers at expiry; precision,
  recall, F1 and confusion matrix. Unknown outcomes are not negatives.
- Chronological origins across all products, with labels available before the
  next evaluation boundary. See the PRD for simulator-only fields excluded from inputs.

Synthetic scenarios deliberately include short-dated and oversized deliveries.
Their prevalence is not a real-world estimate. Successful tests establish data
consistency, not forecasting accuracy or clinical suitability.
