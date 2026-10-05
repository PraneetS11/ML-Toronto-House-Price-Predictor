"""Run with: streamlit run app.py"""

import os
from pathlib import Path

import joblib
import streamlit as st

from house_price.listings import ListingError, fetch_listing, parse_listing_text
from house_price.predict import predict_price

st.set_page_config(page_title="Ontario Home Estimate", page_icon="🏠", layout="centered")
st.markdown("**ONTARIO HOME ESTIMATE** · An applied machine-learning project")
st.title("A starting point for your home search.")
st.write(
    "Paste a Realtor.ca link or describe a home to estimate its listing price from historical Ontario data."
)
st.warning(
    "Historical estimate, not a current appraisal. The source filename indicates 2023 data. "
    "The model learns listing prices, not verified sale prices."
)


@st.cache_resource
def load_model(path, modified):
    # Only load the artifact generated locally by this project's training command.
    return joblib.load(path)


path = Path(os.environ.get("HOUSE_PRICE_MODEL", Path(__file__).parent / "artifacts/model.joblib"))
if not path.exists():
    st.info(
        "The model needs to be trained before estimates are available. "
        "Follow the one-time training step in the README, then reload this page."
    )
    st.stop()
try:
    artifact = load_model(str(path), path.stat().st_mtime_ns)
except Exception:  # noqa: BLE001 - present an actionable message for incompatible local artifacts
    st.error(
        "The saved model could not be loaded. Recreate it using the README training instructions."
    )
    st.stop()

cities = sorted(artifact["city_context"])
for key, default in [("city", "Toronto"), ("bedrooms", 3), ("bathrooms", 2.0)]:
    if key not in st.session_state:
        st.session_state[key] = default


def populate(features):
    for key, value in features.items():
        if key == "city" and value in cities:
            st.session_state[key] = value
        elif key == "bedrooms" and float(value).is_integer():
            st.session_state[key] = int(value)
        elif key == "bathrooms":
            st.session_state[key] = float(value)
    st.session_state.pop("estimate", None)
    imported = ", ".join(features) or "none"
    st.info(
        f"Detected fields: {imported}. Check every field below; missing fields keep their previous values."
    )


with st.expander("Import a Realtor.ca listing", expanded=True):
    url = st.text_input("Listing link", placeholder="https://www.realtor.ca/real-estate/…")
    if st.button("Read listing details", use_container_width=True):
        try:
            with st.spinner("Reading public listing details…"):
                populate(fetch_listing(url))
        except ListingError as exc:
            st.info(str(exc))
    st.caption(
        "Import is best effort. Realtor.ca may block automated access or omit readable metadata. "
        "Only city, bedrooms and bathrooms are imported; asking price is never a model input."
    )
    text = st.text_area(
        "Or paste listing text", placeholder="Toronto, Ontario · 3 + 1 Bedrooms · 2 Bathrooms"
    )
    if st.button("Use pasted details"):
        populate(parse_listing_text(text))

st.subheader("Confirm the home details")
st.selectbox("Ontario city", cities, key="city")
left, right = st.columns(2)
left.number_input("Bedrooms", min_value=0, max_value=20, step=1, key="bedrooms")
right.number_input("Bathrooms", min_value=0.0, max_value=20.0, step=0.5, key="bathrooms")
st.caption(
    "City-level location, population and income are filled from training data. "
    "Property size, condition, type and street-level location are unavailable in this model."
)
inputs = (st.session_state.city, st.session_state.bedrooms, st.session_state.bathrooms)
if st.button("Estimate listing price", type="primary", use_container_width=True):
    try:
        st.session_state.estimate = (inputs, predict_price(artifact, *inputs))
    except ValueError as exc:
        st.error(str(exc))
if "estimate" in st.session_state and st.session_state.estimate[0] == inputs:
    st.metric("Estimated historical listing price · CAD", f"${st.session_state.estimate[1]:,.0f}")
    st.caption(
        "Identical city and room counts produce the same estimate. This is a coarse portfolio model."
    )
    if inputs[1] > 8 or inputs[2] > 6:
        st.warning("Unusual room counts may be poorly represented in training data.")

with st.expander("How reliable is this estimate?"):
    metadata = artifact["metadata"]
    score = metadata["test"]
    a, b, c = st.columns(3)
    a.metric("Held-out MAE", f"${score['mae_cad']:,.0f}")
    b.metric("Held-out RMSE", f"${score['rmse_cad']:,.0f}")
    c.metric("Held-out R²", f"{score['r2']:.3f}")
    st.write(
        "MAE is the average absolute error across held-out listings, not an error bound for your home. "
        "Luxury properties, unusual homes and changing markets can produce much larger errors."
    )
    st.write(
        f"Selected model: {metadata['selected_model'].replace('_', ' ')}. "
        "Property addresses were kept in separate training, validation and test groups."
    )
    st.caption("Independent educational project. Not affiliated with or endorsed by REALTOR.ca.")
