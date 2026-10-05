import joblib
import pytest

from house_price.data import FEATURES, NUMERIC, load_data, split_data
from house_price.model import candidates


@pytest.fixture(scope="session")
def dataset():
    return load_data()[0]


@pytest.fixture(scope="session")
def artifact(dataset):
    training, _, _ = split_data(dataset)
    model = candidates()["ridge"].fit(training[FEATURES], training.Price)
    return {
        "pipeline": model,
        "city_context": training.groupby("City")[NUMERIC[2:]].median().to_dict(orient="index"),
        "metadata": {
            "selected_model": "ridge",
            "test": {"mae_cad": 1.0, "rmse_cad": 2.0, "r2": 0.0},
        },
    }


@pytest.fixture
def model_path(artifact, tmp_path, monkeypatch):
    path = tmp_path / "test-model.joblib"
    joblib.dump(artifact, path)
    monkeypatch.setenv("HOUSE_PRICE_MODEL", str(path))
    return path
