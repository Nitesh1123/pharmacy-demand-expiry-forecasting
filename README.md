# Pharmacy Medicine Demand & Expiry-Risk Forecasting

An individual ML project for pharmacy inventory decision support. The intended
outputs are a 30-day product demand forecast and batch expiry-risk alerts.
Outputs are advisory inventory guidance, never medication or treatment advice.

## Current status: Day 3 — November 8, 2026

Implemented: reproducible synthetic data, weekly inventory snapshots, observed
expiry outcomes, validation, point-in-time feature tables, chronological
train/validation/test splits, an executed EDA notebook, leakage tests, baselines
and candidate models compared on validation data, narrow validation tuning, a
single held-out test evaluation with error analysis, a saved demand model, a
Streamlit inference app, and Windows/Linux CI. A deployed live demo is **not
implemented**; the app runs locally.

See the [PRD and schedule](docs/PRD.md), [data contract](data/README.md),
[EDA notebook](notebooks/01_eda.ipynb), and the [Day 2](reports/day2.md),
[Day 3 data](reports/day3.md) and [Day 3 model comparison](reports/day3-models.md) reports, and the
[held-out test evaluation](reports/test-evaluation.md). The full write-up is the
[technical report](docs/technical_report.md). Submission is November 15, 2026;
the individualized invitation specifies the exact time and timezone.

## Setup

Use **Python 3.13**. The dependency file pins the implemented data, testing,
notebook, model-training and Streamlit stack, including transitive dependencies.

Windows PowerShell (activation is not required):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/data_generator.py
.\.venv\Scripts\python.exe -m src.data_quality --report reports/day2-data-quality.json
.\.venv\Scripts\python.exe -m src.features --report reports/day3-splits.json
.\.venv\Scripts\jupyter.exe execute --inplace notebooks/01_eda.ipynb
.\.venv\Scripts\python.exe -m src.train --report reports/day3-validation.json
.\.venv\Scripts\python.exe -m src.final_evaluation --report reports/test-evaluation.json
.\.venv\Scripts\python.exe -m src.inference
.\.venv\Scripts\python.exe -m src.report_figures
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

Linux/macOS with Python 3.13 installed:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/data_generator.py
.venv/bin/python -m src.data_quality --report reports/day2-data-quality.json
.venv/bin/python -m src.features --report reports/day3-splits.json
.venv/bin/jupyter execute --inplace notebooks/01_eda.ipynb
.venv/bin/python -m src.train --report reports/day3-validation.json
.venv/bin/python -m src.final_evaluation --report reports/test-evaluation.json
.venv/bin/python -m src.inference
.venv/bin/python -m src.report_figures
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/streamlit run app/streamlit_app.py
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

Generator configuration comes from CLI arguments. The pipeline and app
do not load `.env`; `.env.example` holds placeholders only.

## Data flow

```text
data_generator.py -> products / batches / sales / daily_sales / inventory_snapshots
                  -> manifest.json (configuration, versions, file hashes)
data_quality.py   -> relational, temporal, balance and outcome checks -> report
features.py       -> demand_features.csv (Sunday origin x product, 30-day target)
                  -> expiry_features.csv (one near-expiry snapshot per batch)
splits.py         -> shared calendar boundaries; targets/outcomes never cross them
notebooks/01_eda  -> quality, demand patterns, label support, leakage analysis

train.py          -> baselines and candidates fit on train, scored on validation only

final_evaluation.py -> locked selections scored once on the test split

inference.py      -> saves the tested demand model; forecasts and expiry guidance
app/streamlit_app -> loads the saved model; advisory review by Sunday date
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
