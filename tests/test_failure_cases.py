"""Bad inputs must give a clear refusal or a sensible result, never a silent wrong answer."""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_generator import generate_dataset
from src.features import build_demand_features
from src.inference import (available_dates, build_artifact, expiry_guidance, forecast_demand,
                           load_artifact, load_tables, save_artifact)

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
SELECTION = {"demand_boosting": {"selected": {"max_iter": 50, "max_leaf_nodes": 7}}}
TEST_REPORT = {"demand": {"models": {"gradient_boosting_tuned": {"wape": 0.05}}},
               "expiry": {"models": {"rule": {"precision": 0.9, "recall": 0.8}}}}
TABLES = ("products", "batches", "sales", "daily_sales", "inventory_snapshots")
SUNDAY = "2023-06-04"


@pytest.fixture(scope="module")
def tables():
    return generate_dataset(seed=7, num_products=16, days=730)


@pytest.fixture(scope="module")
def artifact(tables):
    return build_artifact(build_demand_features(tables), SELECTION, TEST_REPORT)


def write_tables(tables, directory):
    for name in TABLES:
        tables[name].to_csv(directory / f"{name}.csv", index=False)
    return directory


def run_app(monkeypatch, data_dir, model):
    monkeypatch.setenv("PHARMACY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PHARMACY_MODEL_PATH", str(model))
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.mark.parametrize("contents", [b"not a model", b""])
def test_corrupt_artifact_is_refused(tmp_path, contents):
    path = tmp_path / "broken.joblib"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="valid"):
        load_artifact(path)


def test_artifact_of_the_wrong_type_is_refused(tmp_path):
    path = tmp_path / "list.joblib"
    joblib.dump(["not", "a", "dict"], path)
    with pytest.raises(ValueError, match="valid"):
        load_artifact(path)


def test_artifact_from_another_scikit_learn_version_is_refused(artifact, tmp_path):
    path = save_artifact({**artifact, "scikit_learn": "0.0.1"}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="scikit-learn"):
        load_artifact(path)


def test_tables_failing_quality_checks_are_refused(tables, tmp_path):
    write_tables(tables, tmp_path)
    daily = pd.read_csv(tmp_path / "daily_sales.csv").iloc[1:]
    daily.to_csv(tmp_path / "daily_sales.csv", index=False)
    with pytest.raises(ValueError, match="product-day"):
        load_tables(tmp_path)


def test_table_missing_a_column_is_refused(tables, tmp_path):
    write_tables(tables, tmp_path)
    pd.read_csv(tmp_path / "products.csv").drop(columns="category").to_csv(tmp_path / "products.csv", index=False)
    with pytest.raises((KeyError, ValueError)):
        load_tables(tmp_path)


def test_unseen_category_still_forecasts(tables, artifact):
    changed = {**tables, "products": tables["products"].assign(category="New_Category")}
    result = forecast_demand(artifact, changed, SUNDAY)
    assert np.isfinite(result.forecast_30d).all() and (result.forecast_30d >= 0).all()


def test_extreme_sales_give_a_finite_non_negative_forecast(tables, artifact):
    daily = tables["daily_sales"].assign(quantity_sold=lambda d: d.quantity_sold * 1000)
    result = forecast_demand(artifact, {**tables, "daily_sales": daily}, SUNDAY)
    assert np.isfinite(result.forecast_30d).all() and (result.forecast_30d >= 0).all()


def test_product_with_no_sales_gets_a_low_forecast(tables, artifact):
    daily = tables["daily_sales"].copy()
    product = tables["products"].product_id[0]
    daily.loc[daily.product_id == product, "quantity_sold"] = 0
    result = forecast_demand(artifact, {**tables, "daily_sales": daily}, SUNDAY).set_index("product_id")
    others = result.drop(index=product).forecast_30d
    assert 0 <= result.forecast_30d[product] < others.median()


def test_product_with_no_stock_is_flagged_for_reorder(tables, artifact):
    product = tables["products"].product_id[0]
    snapshots = tables["inventory_snapshots"]
    no_stock = snapshots[~((snapshots.product_id == product) & (snapshots.as_of_date == SUNDAY))]
    result = forecast_demand(artifact, {**tables, "inventory_snapshots": no_stock}, SUNDAY).set_index("product_id")
    assert result.on_hand_units[product] == 0
    assert result.action[product] == "Review reorder"
    assert result.shortfall_units[product] == result.forecast_30d[product]


def test_date_with_no_near_expiry_batches_returns_an_empty_table(tables):
    batches = tables["batches"].assign(expiry_date="2030-01-01")
    result = expiry_guidance({**tables, "batches": batches}, SUNDAY)
    assert result.empty and not result.alert.any()


def test_history_too_short_for_any_review_date_shows_a_message(monkeypatch, artifact, tmp_path):
    short = generate_dataset(seed=7, num_products=4, days=30)
    assert available_dates(short) == []
    model = save_artifact(artifact, tmp_path / "model.joblib")
    app = run_app(monkeypatch, write_tables(short, tmp_path), model)
    assert not app.exception
    assert any("56 days" in e.value for e in app.error)


def test_corrupt_data_shows_a_message_in_the_app(monkeypatch, tables, artifact, tmp_path):
    write_tables(tables, tmp_path)
    (tmp_path / "batches.csv").write_text("", encoding="utf-8")
    model = save_artifact(artifact, tmp_path / "model.joblib")
    app = run_app(monkeypatch, tmp_path, model)
    assert not app.exception
    assert app.error


def test_corrupt_model_shows_a_message_in_the_app(monkeypatch, tables, tmp_path):
    write_tables(tables, tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"not a model")
    app = run_app(monkeypatch, tmp_path, tmp_path / "model.joblib")
    assert not app.exception
    assert any("python -m src.inference" in e.value for e in app.error)


@pytest.mark.parametrize("module", ["src.features", "src.train", "src.final_evaluation", "src.inference"])
def test_command_line_tools_explain_missing_data(module, tmp_path):
    result = subprocess.run([sys.executable, "-m", module, "--data-dir", str(tmp_path)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
