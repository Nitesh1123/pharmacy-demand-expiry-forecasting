from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data_generator import generate_dataset
from src.features import build_demand_features
from src.inference import (available_dates, build_artifact, forecast_demand, load_artifact,
                           save_artifact)

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
SELECTION = {"demand_boosting": {"selected": {"max_iter": 50, "max_leaf_nodes": 7}}}
TEST_REPORT = {"demand": {"models": {"gradient_boosting_tuned": {"wape": 0.05}}},
               "expiry": {"models": {"rule": {"precision": 0.9, "recall": 0.8}}}}


@pytest.fixture(scope="module")
def tables():
    return generate_dataset(seed=7, num_products=16, days=730)


@pytest.fixture(scope="module")
def paths(tables, tmp_path_factory):
    root = tmp_path_factory.mktemp("app")
    for name in ("products", "batches", "daily_sales", "inventory_snapshots"):
        tables[name].to_csv(root / f"{name}.csv", index=False)
    return root, save_artifact(build_artifact(build_demand_features(tables), SELECTION, TEST_REPORT),
                               root / "model.joblib")


def metric(app, label):
    return next(m.value for m in app.metric if m.label == label)


def run_app(monkeypatch, data_dir, model_path):
    monkeypatch.setenv("PHARMACY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PHARMACY_MODEL_PATH", str(model_path))
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_shows_the_saved_model_forecast_and_advisory_notice(monkeypatch, paths, tables):
    root, model = paths
    app = run_app(monkeypatch, root, model)
    assert not app.exception
    assert any("not medication or treatment advice" in w.value for w in app.warning)
    latest = available_dates(tables)[-1]
    expected = forecast_demand(load_artifact(model), tables, latest)
    first = expected.iloc[0]
    assert app.selectbox(key="review_date").value == latest
    assert metric(app, "30-day forecast") == f"{first.forecast_30d:,.0f} units"


def test_changing_the_review_date_updates_the_forecast(monkeypatch, paths, tables):
    root, model = paths
    app = run_app(monkeypatch, root, model)
    before = metric(app, "30-day forecast")
    app.selectbox(key="review_date").set_value(available_dates(tables)[0]).run()
    assert not app.exception
    assert metric(app, "30-day forecast") != before


def test_missing_model_shows_build_instructions_instead_of_crashing(monkeypatch, paths):
    root, _ = paths
    app = run_app(monkeypatch, root, root / "absent.joblib")
    assert not app.exception
    assert any("python -m src.inference" in e.value for e in app.error)


def test_missing_data_shows_build_instructions_instead_of_crashing(monkeypatch, paths, tmp_path):
    _, model = paths
    app = run_app(monkeypatch, tmp_path, model)
    assert not app.exception
    assert any("python src/data_generator.py" in e.value for e in app.error)
