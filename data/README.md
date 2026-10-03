# Dataset provenance

`raw/ontario_listings_source.csv` is the original
`HouseListings-Top45Cities-10292023-kaggle.csv`, moved without modifying its bytes.
It contains 35,768 records from Canadian cities, including non-Ontario listings.
The filename suggests an October 29, 2023 snapshot and Kaggle provenance, but the
original repository does not record a dataset URL, collector, license, currency
metadata, or collection methodology. CAD and listing/asking prices are working
interpretations, not independently verified transaction labels. Confirm source
and redistribution rights before republishing this dataset elsewhere.

The loader explicitly records and skips malformed-width records (source lines 2
and 3); it never allows pandas to silently shift columns. It keeps a fixed allowlist
of Ontario cities, requires positive finite prices and a usable address for group
splitting, removes exact duplicate records, and rejects invalid room counts.
Missing numeric features are imputed using training data only. It does not trim
price tails. Address is used only to group repeated properties, never as a predictor.
See `reports/metrics.json` for exact audit counts after running training.
