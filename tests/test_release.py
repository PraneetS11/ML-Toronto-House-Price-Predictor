"""Check the actual versioned deployment model, independently of UI fixtures."""

import json
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest

from house_price.predict import predict_price


def test_release_model_matches_reports():
    artifact = joblib.load("artifacts/model.joblib")
    report = json.loads(Path("reports/metrics.json").read_text())
    assert artifact["metadata"] == report
    assert 1_102_603 < predict_price(artifact, "Toronto", 3, 2) < 1_102_605


def test_release_starts_without_training(monkeypatch):
    monkeypatch.delenv("HOUSE_PRICE_MODEL", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    next(button for button in app.button if button.label == "Estimate listing price").click().run()
    assert not app.exception
    assert app.metric[0].value == "$1,102,604"
