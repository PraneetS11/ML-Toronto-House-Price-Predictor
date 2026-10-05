from pathlib import Path

from streamlit.testing.v1 import AppTest


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_manual_estimate_and_stale_result(model_path):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    button(app, "Estimate listing price").click().run()
    assert not app.exception
    assert app.metric[0].label == "Estimated historical listing price · CAD"
    app.number_input[0].set_value(4).run()
    assert all(item.label != "Estimated historical listing price · CAD" for item in app.metric)


def test_pasted_listing_details(model_path):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.text_area[0].set_value("Hamilton, Ontario · 4 Bedrooms · 3 Bathrooms")
    button(app, "Use pasted details").click().run()
    assert not app.exception
    assert app.selectbox[0].value == "Hamilton"
    assert app.number_input[0].value == 4
    button(app, "Estimate listing price").click().run()
    assert not app.exception


def test_bad_link_has_actionable_fallback(model_path):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.text_input[0].set_value("https://example.com")
    button(app, "Read listing details").click().run()
    assert not app.exception
    assert any("HTTPS Realtor.ca" in item.value for item in app.info)


def test_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("HOUSE_PRICE_MODEL", str(tmp_path / "missing.joblib"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    assert any("trained" in item.value for item in app.info)
