"""Validated inference using the same serialized preprocessing pipeline."""

import math

import pandas as pd

from house_price.data import FEATURES


def predict_price(artifact, city, bedrooms, bathrooms):
    if city not in artifact["city_context"]:
        raise ValueError("Choose a supported Ontario city; other locations were not modeled.")
    for label, value in [("Bedrooms", bedrooms), ("Bathrooms", bathrooms)]:
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 20:
            raise ValueError(f"{label} must be a finite number between 0 and 20.")
    if float(bedrooms) != int(bedrooms):
        raise ValueError("Bedrooms must be a whole number.")
    if bathrooms * 2 != int(bathrooms * 2):
        raise ValueError("Bathrooms must use half-bath increments.")
    row = {
        "City": city,
        "Number_Beds": bedrooms,
        "Number_Baths": bathrooms,
        **artifact["city_context"][city],
    }
    value = float(artifact["pipeline"].predict(pd.DataFrame([row])[FEATURES])[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("This combination produced an invalid estimate. Try a typical home.")
    return value
