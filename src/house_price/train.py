"""Run a fixed experiment, select on validation MAE, evaluate once on test."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from house_price.data import DEFAULT_DATA, FEATURES, NUMERIC, load_data, split_data
from house_price.model import candidates


def metrics(actual, predicted):
    return {
        "mae_cad": float(mean_absolute_error(actual, predicted)),
        "rmse_cad": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def train(data=DEFAULT_DATA, output=Path("artifacts"), reports=Path("reports"), seed=42):
    output, reports = Path(output), Path(reports)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame, audit = load_data(data)
    training, validation, test = split_data(frame, seed)
    fitted, scores = {}, {}
    for name, pipeline in candidates(seed).items():
        pipeline.fit(training[FEATURES], training.Price)
        scores[name] = metrics(validation.Price, pipeline.predict(validation[FEATURES]))
        fitted[name] = pipeline
    winner = min(scores, key=lambda name: scores[name]["mae_cad"])
    pipeline = fitted[winner]
    predictions = pipeline.predict(test[FEATURES])
    # Keep the selected model fitted on train only, exactly as evaluated on validation.
    city_context = training.groupby("City")[NUMERIC[2:]].median().to_dict(orient="index")
    result = {
        "seed": seed,
        "dataset": audit,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "split_rows": {
            k: len(v) for k, v in [("train", training), ("validation", validation), ("test", test)]
        },
        "split_groups": {
            k: int(v.property_group.nunique())
            for k, v in [("train", training), ("validation", validation), ("test", test)]
        },
        "selection_metric": "validation MAE",
        "selected_model": winner,
        "validation": scores,
        "test": metrics(test.Price, predictions),
        "test_median_baseline": metrics(
            test.Price, fitted["median_baseline"].predict(test[FEATURES])
        ),
        "test_by_city": {
            city: {
                "rows": int((test.City == city).sum()),
                **metrics(test.loc[test.City == city, "Price"], predictions[test.City == city]),
            }
            for city in sorted(test.City.unique())
        },
        "training_price_summary": training.Price.describe().to_dict(),
    }
    artifact = {
        "pipeline": pipeline,
        "city_context": city_context,
        "metadata": result,
        "schema_version": 1,
    }
    joblib.dump(artifact, output / "model.joblib", compress=3)
    (reports / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # Only training data informs EDA. Test scatter is a final diagnostic, not tuning input.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(np.log10(training.Price), bins=40, color="#227c9d")
    axes[0].set(
        xlabel="log10 listing price (CAD assumed)",
        ylabel="Training listings",
        title="Training price distribution — no tail trimming",
    )
    training.groupby("City").Price.median().sort_values().plot.barh(ax=axes[1], color="#227c9d")
    axes[1].set(xlabel="Median training listing price (CAD)", title="Ontario city differences")
    fig.savefig(reports / "training_eda.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.scatter(test.Price, predictions, alpha=0.25, s=12, color="#227c9d")
    top = max(test.Price.max(), predictions.max())
    ax.plot([0, top], [0, top], "--", color="#666666")
    ax.set(
        xlabel="Observed listing price (CAD)",
        ylabel="Predicted price (CAD)",
        title=f"Held-out properties · {winner}",
    )
    fig.savefig(reports / "test_predictions.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "selected_model": winner,
                "split_rows": result["split_rows"],
                "validation": scores,
                "test": result["test"],
            },
            indent=2,
        )
    )
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.reports, args.seed)


if __name__ == "__main__":
    main()
