import csv

import joblib
import numpy as np
import pytest

from house_price.data import COLUMNS, FEATURES, load_data, split_data
from house_price.model import candidates
from house_price.predict import predict_price


def test_source_audit():
    frame, audit = load_data()
    assert audit["malformed_lines"] == [2, 3]
    assert audit["retained_rows"] == 13828
    assert audit["exact_duplicates_removed"] == 163
    assert frame.Price.min() > 0
    assert "Winnipeg" not in frame.City.values


def test_group_splits_disjoint_and_repeatable(dataset):
    first = split_data(dataset)
    second = split_data(dataset)
    groups = [set(part.property_group) for part in first]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sum(map(len, first)) == len(dataset)
    for left, right in zip(first, second):
        assert left.index.tolist() == right.index.tolist()


def test_schema_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Price,City\n100,Toronto\n")
    with pytest.raises(ValueError, match="schema"):
        load_data(path)


def test_invalid_target_and_missing_feature(tmp_path):
    path = tmp_path / "sample.csv"
    rows = [
        ["Toronto", 100000 + i, f"{i} Main St", "", 2, "Ontario", 100000, 43, -79, 90000]
        for i in range(40)
    ]
    rows[0][1] = -1
    with path.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    frame, audit = load_data(path)
    assert audit["invalid_rows"] == 1
    assert frame.Number_Beds.isna().all()


def test_preprocessing_does_not_learn_from_inference(dataset):
    training, validation, _ = split_data(dataset)
    model = candidates()["ridge"].fit(training[FEATURES], training.Price)
    transformer = model.named_steps["preprocess"].named_transformers_["log_numeric"]
    before = transformer.named_steps["impute"].statistics_.copy()
    new = validation[FEATURES].head(2).copy()
    new["City"] = "Unseen city"
    new["Number_Beds"] = np.nan
    assert np.isfinite(model.predict(new)).all()
    np.testing.assert_array_equal(before, transformer.named_steps["impute"].statistics_)
    assert "Price" not in model.feature_names_in_
    assert "Address" not in model.feature_names_in_


def test_serialization_preserves_predictions(artifact, model_path):
    before = predict_price(artifact, "Toronto", 3, 2)
    after = predict_price(joblib.load(model_path), "Toronto", 3, 2)
    assert before == pytest.approx(after)


@pytest.mark.parametrize(
    "city,beds,baths",
    [
        ("Vancouver", 3, 2),
        ("Toronto", -1, 2),
        ("Toronto", 2.5, 2),
        ("Toronto", 3, float("nan")),
        ("Toronto", 3, 21),
        ("Toronto", 3, 1.2),
    ],
)
def test_invalid_inference_inputs(artifact, city, beds, baths):
    with pytest.raises(ValueError):
        predict_price(artifact, city, beds, baths)
