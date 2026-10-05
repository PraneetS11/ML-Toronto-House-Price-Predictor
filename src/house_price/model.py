"""All learned preprocessing stays inside each sklearn pipeline."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

LOG_FEATURES = ["Number_Beds", "Number_Baths", "Population"]
OTHER_FEATURES = ["Latitude", "Longitude", "Median_Family_Income"]


def preprocessing():
    log_numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer(
        [
            ("log_numeric", log_numeric, LOG_FEATURES),
            ("numeric", numeric, OTHER_FEATURES),
            ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City"]),
        ]
    )


def candidates(seed=42):
    estimators = {
        "median_baseline": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=10),
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=5, max_depth=16, random_state=seed, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=10,
            early_stopping=False,
            random_state=seed,
        ),
    }
    return {
        name: Pipeline([("preprocess", preprocessing()), ("model", estimator)])
        for name, estimator in estimators.items()
    }
