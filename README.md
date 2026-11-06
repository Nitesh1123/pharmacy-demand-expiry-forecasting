# Pharmacy Medicine Demand & Expiry-Risk Forecasting

Decision-support Machine Learning system for retail pharmacies:
1. **Demand Forecasting (Regression):** Multi-horizon product-level demand prediction using chronological train/validation/test splits.
2. **Expiry-Risk Flagging (Classification):** Predictive identification of batch lots at risk of expiring unsold before their expiry date.

> **Disclaimer:** This tool is strictly intended for **inventory operational decision-support** (stock rotation, markdown, reordering). It does not provide clinical, diagnostic, or treatment advice.

---

## Architecture Overview

```
                      ┌────────────────────────┐
                      │  src/data_generator.py │
                      └───────────┬────────────┘
                                  │ (products, batches, sales)
                                  ▼
                      ┌────────────────────────┐
                      │  01_eda.ipynb          │
                      │  (stationarity, ACF)   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │  src/features.py       │
                      │  (leakage-safe lags)   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │  src/train.py          │
                      │  (baseline vs models)  │
                      └───────────┬────────────┘
                                  │ (models/*.joblib)
                                  ▼
                 ┌───────────────────────────────────┐
                 │  Inference Services               │
                 │  - FastAPI (app/api.py)           │
                 │  - Streamlit (app/streamlit_app)  │
                 └───────────────────────────────────┘
```

---

## Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2. Generate Synthetic Data
```bash
python src/data_generator.py
```

### 3. Run Training Pipeline
```bash
python src/train.py
```

### 4. Launch Inference Interfaces
- **FastAPI Model Server:**
  ```bash
  uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Interactive Streamlit App:**
  ```bash
  streamlit run app/streamlit_app.py
  ```

---

## Evaluation Metrics

- **Demand Regression:** WAPE (Weighted Absolute Percentage Error), MAE, RMSE
- **Expiry Classification:** Precision, Recall, F1-Score, Confusion Matrix
- **Interpretability:** SHAP feature attribution summary
