# Original experiment

`original_house_price_predictor.ipynb` is preserved byte-for-byte from commit
`5a06ba4`. It demonstrates the original Ontario filtering, one-hot encoding,
log transforms, correlation/distribution EDA, linear regression, random forest,
and grid search. It is an archival reference, not the supported execution path.

Known issues: source CSV has malformed records; price quantiles were computed
before splitting; splits and forests had no fixed seed; test scores were inspected
for several models; `best_forest` was never assigned. Saved scores are historical
outputs, not verified results of the revised pipeline. See the root README.

The empty notebook checkpoint was removed. `ca_real_estate.csv` was unused by the
notebook and had inconsistent city/province pairs; it was removed from the active
project, without claiming an unverified origin. Both remain in Git history.
