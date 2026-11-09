"""Advisory pharmacy stock review, served from the saved demand model."""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import (MODEL_PATH, available_dates, expiry_guidance, explain_forecast,  # noqa: E402
                           forecast_demand, load_artifact, load_tables)

DATA_DIR = os.environ.get("PHARMACY_DATA_DIR", str(ROOT / "data"))
MODEL_FILE = os.environ.get("PHARMACY_MODEL_PATH", str(ROOT / MODEL_PATH))


@st.cache_resource
def cached_artifact(path):
    return load_artifact(path)


@st.cache_data
def cached_tables(path):
    return load_tables(path)


@st.cache_data
def cached_forecast(_artifact, _tables, model, data, as_of):
    return forecast_demand(_artifact, _tables, as_of)


@st.cache_data
def cached_expiry(_tables, data, as_of):
    return expiry_guidance(_tables, as_of)


def readable(value):
    return f"{value:,.2f}" if isinstance(value, (int, float)) or hasattr(value, "item") else str(value)


st.set_page_config(page_title="Pharmacy Stock Review", layout="wide")
# Captions carry the caveats, so keep them readable on the soft theme backgrounds.
st.html('<style>.stApp [data-testid="stCaptionContainer"] { opacity: 0.8; }</style>')
st.title("Pharmacy stock review")
st.warning("Advisory stock guidance for human review, not medication or treatment advice. "
           "Results come from synthetic data and do not show real-pharmacy accuracy.")

try:
    tables = cached_tables(DATA_DIR)
except (FileNotFoundError, KeyError, ValueError):
    st.error("Data is missing or failed its quality checks. Run `python src/data_generator.py`, "
             "then `python -m src.features`.")
    st.stop()
try:
    artifact = cached_artifact(MODEL_FILE)
except (FileNotFoundError, ValueError):
    st.error("Saved model not found or not valid. Run `python -m src.inference` to build it.")
    st.stop()

dates = available_dates(tables)
if not dates:
    st.error("Not enough history to forecast: the data needs at least 56 days of daily sales.")
    st.stop()
review_date = st.sidebar.selectbox("Review date (end of Sunday)", dates, index=len(dates) - 1,
                                   key="review_date")
st.sidebar.caption("Forecasts use only data recorded through the end of this date.")

forecast = cached_forecast(artifact, tables, MODEL_FILE, DATA_DIR, review_date)
expiry = cached_expiry(tables, DATA_DIR, review_date)
demand_tab, expiry_tab, about_tab = st.tabs(["Demand forecast", "Expiry alerts", "About this model"])

with demand_tab:
    left, middle = st.columns(2)
    left.metric("Products to review for reorder", int((forecast.shortfall_units > 0).sum()))
    middle.metric("Forecast units, all products", f"{forecast.forecast_30d.sum():,.0f}")
    st.subheader("Next 30 days by product")
    st.dataframe(
        forecast.sort_values("shortfall_units", ascending=False)[
            ["product_id", "product_name", "category", "forecast_30d", "on_hand_units",
             "shortfall_units", "action"]],
        hide_index=True, width="stretch",
        column_config={"product_id": "Product", "product_name": "Name", "category": "Category",
                       "forecast_30d": st.column_config.NumberColumn("30-day forecast", format="%.0f"),
                       "on_hand_units": "On hand",
                       "shortfall_units": st.column_config.NumberColumn("Shortfall", format="%.0f"),
                       "action": "Suggested review"})

    st.subheader("Explain one forecast")
    names = dict(zip(forecast.product_id, forecast.product_name))
    product = st.selectbox("Product", forecast.product_id.tolist(), key="product",
                           format_func=lambda p: f"{p}: {names[p]}")
    row = forecast[forecast.product_id == product].iloc[0]
    first, second, third = st.columns(3)
    first.metric("30-day forecast", f"{row.forecast_30d:,.0f} units")
    second.metric("Typical error range", f"{row.low_30d:,.0f} to {row.high_30d:,.0f} units")
    third.metric("Stock on hand", f"{row.on_hand_units:,.0f} units")
    st.caption(f"The range applies the model's {artifact['test_wape']:.1%} WAPE from the held-out "
               "test set. It's a rough guide, not a guaranteed interval.")
    if pd.notna(row.recorded_30d):
        st.info(f"Historical check: {row.recorded_30d:,.0f} units were actually recorded in "
                "these 30 days. The model did not see this value.")
    drivers = pd.DataFrame(explain_forecast(artifact, tables, review_date, product))
    drivers = pd.DataFrame({
        "Input": drivers.feature.str.replace("_", " "),
        "This product": drivers.value.map(readable),
        "Typical in training": drivers.typical_train_value.map(readable),
        "Effect on forecast (units)": drivers.prediction_change_from_actual_value.round(0)})
    st.caption("Largest effects, from resetting one input at a time to its typical training "
               "value. These describe the model's behaviour and are not causes of demand.")
    st.dataframe(drivers, hide_index=True, width="stretch")

with expiry_tab:
    st.metric("Batches flagged", int(expiry.alert.sum()), help="Out of batches within 60 days of expiry")
    st.caption(f"{len(expiry)} batches are within 60 days of expiry on this date. A batch is flagged "
               "when projected unsold stock is over 20% of the quantity received.")
    st.dataframe(
        expiry, hide_index=True, width="stretch",
        column_order=["batch_id", "product_id", "category", "expiry_date", "days_to_expiry",
                      "quantity_remaining", "units_ahead_fefo", "mean_28d",
                      "projected_leftover_fraction", "action"],
        column_config={
            "batch_id": "Batch", "product_id": "Product", "category": "Category",
            "expiry_date": "Expires", "days_to_expiry": "Days left", "quantity_remaining": "Units left",
            "units_ahead_fefo": "Units expiring first",
            "mean_28d": st.column_config.NumberColumn("Sales per day (28d)", format="%.1f"),
            "projected_leftover_fraction": st.column_config.NumberColumn("Projected unsold share",
                                                                         format="percent"),
            "action": "Suggested review"})

with about_tab:
    st.markdown(f"""
**Demand model:** gradient boosting loaded from the saved artifact, fitted on data through
the {artifact['trained_through']} origin. Settings: {artifact['params']}.

**Expiry alerts:** a transparent rule. Projected sellable units are the 28-day sales rate
times days left, minus units that expire first. It is a rule, not a trained model.

**Held-out test results (synthetic data):** demand WAPE {artifact['test_wape']:.1%}.
Expiry alert precision {artifact['test_expiry_rule']['precision']:.2f}, recall {artifact['test_expiry_rule']['recall']:.2f}.

**Known limits**
- Data is synthetic, with short-dated and oversized deliveries made deliberately common.
- There are no real stockouts, lead times, returns or promotions.
- The expiry rule can raise false alarms when queued stock expires before demand reaches it.
- It can miss batches when a later delivery with an earlier expiry takes the demand.
- Stock on hand includes units that may expire within the 30-day window.
- A pharmacist or inventory manager should review every suggestion.
""")
