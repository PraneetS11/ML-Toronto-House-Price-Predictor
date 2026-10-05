"""Auditable ingestion and property-grouped splits."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CITIES = (
    "Barrie",
    "Brantford",
    "Caledon",
    "Guelph",
    "Hamilton",
    "Kingston",
    "Kitchener",
    "London",
    "Oshawa",
    "Ottawa",
    "Peterborough",
    "Sault Ste. Marie",
    "St. Catharines",
    "Sudbury",
    "Thunder Bay",
    "Toronto",
    "Windsor",
)
NUMERIC = [
    "Number_Beds",
    "Number_Baths",
    "Population",
    "Latitude",
    "Longitude",
    "Median_Family_Income",
]
FEATURES = ["City", *NUMERIC]
COLUMNS = [
    "City",
    "Price",
    "Address",
    "Number_Beds",
    "Number_Baths",
    "Province",
    "Population",
    "Latitude",
    "Longitude",
    "Median_Family_Income",
]
DEFAULT_DATA = Path("data/raw/ontario_listings_source.csv")


def load_data(path=DEFAULT_DATA):
    """Skip malformed records explicitly; apply fixed, target-independent scope rules."""
    path = Path(path)
    rows, malformed = [], []
    with path.open(encoding="latin1", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != COLUMNS:
            raise ValueError("Unexpected CSV schema; expected original Canadian listings columns.")
        for line, row in enumerate(reader, 2):
            if len(row) != len(COLUMNS):
                malformed.append(line)
            else:
                rows.append(row)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    audit = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(rows) + len(malformed),
        "malformed_lines": malformed,
    }
    scope = (frame.Province == "Ontario") & frame.City.isin(CITIES)
    audit["outside_scope_rows"] = int((~scope).sum())
    frame = frame.loc[scope].copy()
    for col in ["Price", *NUMERIC]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = frame.Price.gt(0) & frame.Address.str.strip().ne("")
    for col in ["Number_Beds", "Number_Baths"]:
        valid &= frame[col].isna() | frame[col].between(0, 20)
    audit["invalid_rows"] = int((~valid).sum())
    frame = frame.loc[valid].copy()
    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    audit["exact_duplicates_removed"] = before - len(frame)
    # Keep unit identifiers; normalize punctuation and whitespace for repeated addresses.
    frame["property_group"] = (
        frame.City + "|" + frame.Address.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    audit["retained_rows"] = len(frame)
    audit["unique_properties"] = int(frame.property_group.nunique())
    if len(frame) < 30:
        raise ValueError("At least 30 usable records are required for grouped evaluation.")
    return frame, audit


def split_data(frame, seed=42):
    """Approximately 60/20/20 by property groups, never by individual listing row."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    development, test = next(splitter.split(frame, groups=frame.property_group))
    dev = frame.iloc[development]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train, validation = next(splitter.split(dev, groups=dev.property_group))
    return dev.iloc[train].copy(), dev.iloc[validation].copy(), frame.iloc[test].copy()
