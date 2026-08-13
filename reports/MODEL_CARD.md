# Model card — HomeCast Gurgaon valuation model

## Overview

Gradient-boosted regression tree ensemble that estimates a residential
property's asking price in Gurgaon, India, from a handful of listing
attributes. Trained and evaluated on a single snapshot of scraped listing
data — see [Limitations](#limitations).

## Data provenance and snapshot nature

- Source: `data/gurgaon/raw/gurgaon_properties.csv`, 3,803 residential listings
  (flats and independent houses / builder floors) collected from a public
  Indian property-listing portal at one point in time. **Which portal is not
  recorded anywhere in this repository, so it is not named here — a guess
  would be worse than the gap.** The column set (`areaWithType`,
  `agePossession`, `luxury_score`) is characteristic of a listings scrape, but
  nothing in the file, its metadata, or the pipeline identifies the site or
  the collection date.
  There is no listing date column and no repeat observations of the same
  property over time — this is a cross-sectional snapshot, not a time series.
- **Licensing.** The `LICENSE` file (MIT) covers the *code* in this repository
  only. The committed CSV is third-party listing content, redistributed here
  so the published numbers can be reproduced. It carries no licence grant from
  this project; check the originating portal's terms before using it for
  anything beyond study and reproduction.
- After cleaning (see `data/DATA_DICTIONARY.md` and the README's Methodology
  section): 3,600 listings, 2,799 flats and 801 houses.
- All prices are **asking prices** as listed on the portal, not confirmed
  transaction prices.

## Target

`log(price)`, where `price` is the listing's asking price in ₹ crore. The
model is trained in log space (right-skewed price distribution: mean ₹2.51
Cr, median ₹1.54 Cr, max ₹31.5 Cr) and predictions are exponentiated back to
₹ crore for reporting.

## Features

| Feature | Description |
|---|---|
| `area` | Listing area, sq. ft. |
| `bedrooms` | Bedroom count |
| `bathrooms` | Bathroom count |
| `is_house` | 1 if independent house/builder floor, 0 if flat |
| `furnishing_code` | `unfurnished`=0, `semi-furnished`=1, `furnished`=2 |
| `luxury_score` | Portal-derived amenity score |
| `age_code` | `Under Construction`=0 … `Old Property`=4, missing=-1 |
| `sector_ppsf` | Sector target-encoded as median ₹/sqft (see below) |

Feature importances from the final trained model: `area` 0.537, `sector_ppsf`
0.195, `bathrooms` 0.157, `is_house` 0.085 (the remaining four features share
0.027 between them). Area and the sector encoding together account for 73.2%
of the model's decisions. These come from
`GradientBoostingRegressor.feature_importances_` and are exported verbatim to
`models/gurgaon/model.json` under `feature_importances`; the dashboard reads
them from the payload, and `tests/test_export.py` asserts the exported values
equal the fitted model's.

## Fold-safe sector encoding

`sector_ppsf` is a target encoding — the sector's median ₹/sqft — and is the
single most informative categorical feature. Computing it on the full
dataset before splitting into train/test would leak each listing's own price
(and its sector-mates' prices) into a feature used to predict that same
price, inflating the apparent accuracy. To avoid this, `sector_ppsf` is
**re-learned from scratch inside every cross-validation training fold**,
using only that fold's training rows, then applied to the held-out fold.
This is implemented in `src/homecast/features.py:sector_encoding` and called
once per fold in `src/homecast/model.py:evaluate`.

## Model parameters

`sklearn.ensemble.GradientBoostingRegressor`, with HomeCast's own project
defaults (`DEFAULT_PARAMS` in `src/homecast/model.py` — not scikit-learn's
library defaults), chosen up front and never tuned against the results:

```
n_estimators   = 300
max_depth      = 3
learning_rate  = 0.06
subsample      = 0.9
random_state   = 7
```

No hyperparameter search was run. These defaults beat both baselines below
on the first attempt, so no tuning grid was needed.

## Reproducing these numbers

Every figure in this card was produced on Python 3.12.13 with pandas 3.0.3,
NumPy 2.5.1, scikit-learn 1.9.0, SciPy 1.18.0, joblib 1.5.3 and Matplotlib
3.11.0 — the pinned set in `requirements.txt`. Install that file before
`pip install -e .`. The pipeline is fully seeded (`random_state=7` for both
the model and the `KFold` split), so those versions reproduce the metrics
exactly; other versions may differ in the last decimal (an older scikit-learn
gives R² 0.808 rather than 0.806).

## Cross-validation protocol

5-fold `KFold(shuffle=True, random_state=7)` over all 3,600 cleaned
listings. Predictions are collected out-of-fold — every metric below is
computed on rows the model never saw during training for that fold. The
final shipped model is then refit on all 3,600 rows using the same
parameters and a sector encoding learned from the full dataset (only the CV
metrics use fold-local encodings; the deployed model uses one encoding
learned from everything it has).

## Metrics vs. baselines

Out-of-fold, 5-fold CV, n = 3,600:

| Model | MAE (₹ lakh) | MAPE (%) | R² |
|---|---:|---:|---:|
| HomeCast model (GBR) | 54.4 | 21.4 | 0.806 |
| Sector-median ₹/sqft rule | 72.8 | 26.6 | 0.653 |
| Global-median ₹/sqft rule | 114.9 | 38.1 | 0.299 |

Full precision (`models/gurgaon/metrics.json`): model MAE = 54.370871... lakh,
MAPE = 21.388963...%, R² = 0.805567...; sector baseline MAE = 72.786742...
lakh, MAPE = 26.645038...%, R² = 0.652677...; global baseline MAE =
114.936236... lakh, MAPE = 38.078945...%, R² = 0.299244....

Both baselines are the pricing rules an agent uses without a model: multiply
a median ₹/sqft figure (global, or specific to the listing's sector) by the
listing's area. The model beats the sector-median rule by roughly a quarter
on MAE, and the sector-median rule in turn beats the global-median rule —
the expected ordering holds.

## Uncertainty band

`models/gurgaon/model.json` → `band`: `[-0.3127631541941217,
0.31978747923539713]`, the 10th and 90th percentile of the out-of-fold
log-residuals across all 5 folds. The residual is defined as
`log(pred) - log(actual)`, so `actual = pred * exp(-residual)` — **the sign is
negated when the residual band becomes a price band.** The 90th-percentile
residual (the largest overestimates) sets the *low* end of the true price and
the 10th-percentile residual sets the *high* end:

```
lo = pred * exp(-0.31979) = pred * 0.7263   ->  -27.4%
hi = pred * exp(+0.31276) = pred * 1.3672   ->  +36.7%
```

So the multiplicative range around a point estimate is **-27.4% to +36.7%**,
and the true price is expected to fall inside it about 80% of the time.

The percentage range is lopsided, but the band underneath it is
near-symmetric in log space: q10 is -0.313 and q90 is +0.320, a gap of under
0.008 log units. Exponentiating a symmetric log interval always produces a
larger positive percentage than negative one (`exp(x) - 1 > |exp(-x) - 1|`
for any `x > 0`), so the -27%/+37% shape is arithmetic and would look the same
for perfectly symmetric errors. It is not evidence that the model's misses
skew one way. The measured residual skew is in fact slightly negative
(-0.269), which points the other way; it is small enough that the honest
reading is "no meaningful asymmetry in the band itself". Where the errors
genuinely are uneven is across the price range — see the quintile structure
below.

## Error structure by price quintile

Mean signed log-residual by price quintile (positive = model overestimates):

| Quintile | Mean signed log-residual |
|---|---:|
| Q1 (cheapest) | +0.129 |
| Q2 | +0.033 |
| Q3 | −0.001 |
| Q4 | −0.033 |
| Q5 (priciest) | −0.129 |

Monotonic across quintiles: the model **overestimates the cheapest listings
and underestimates the most expensive ones**. 42% of the largest
overestimates fall in Q1 and 44% of the largest underestimates fall in Q5,
against a 20% base rate each. Q1 also carries the highest mean absolute
percentage error of the five quintiles, at 27.6% — this is the segment where
the model is least reliable.

## Intended use

- Estimating a defensible asking-price range for a Gurgaon flat or house
  given its sector, size, bedroom/bathroom count, furnishing, luxury score,
  and (optionally) age/possession status.
- Benchmarking a specific listing or valuation against what a sector-median
  ₹/sqft rule alone would produce.
- Not intended for: valuing properties outside Gurgaon (no other city's data
  is in the training set), pricing outside the model's trained range (see
  below), or substituting for a professional appraisal, legal title check,
  or lender valuation.

## Limitations

- **Asking prices, not transaction prices.** The training data is listing
  prices scraped from a portal. Actual sale prices — after negotiation — are
  not observed. The model learns what sellers ask, not necessarily what
  buyers pay.
- **Single snapshot, no time dimension.** All listings come from one point
  in time with no listing date. The model cannot account for market
  movement before or after that snapshot, and should be retrained on fresh
  data before being trusted for current pricing.
- **No geographic coordinates.** Location is represented only by sector
  (`sector_ppsf`), a discrete label with a target-encoded median. Two
  listings in the same sector but very different micro-locations (adjacent
  to a metro stop vs. a back lane) get the same location signal.
- **Out-of-range inputs are rejected, not extrapolated.** `homecast predict`
  rejects an `area`, `bedrooms`, `bathrooms`, or `luxury_score` outside the
  range seen in training (`fitted.ranges`, from the min/max of the cleaned
  dataset) rather than silently extrapolating a tree-based model past its
  training distribution, where gradient-boosted trees produce flat,
  unreliable predictions. An unknown `sector`, `property_type`, `furnishing`,
  or age label is rejected too, so a typo cannot quietly collapse to a
  default. What is *not* guarded is an unusual **combination** of individually
  in-range inputs — a 1-BHK with a top luxury score, say — which still returns
  a confident-looking number the model has little basis for.
- **Weakest at the low end.** Q1 (the cheapest fifth of listings) has both
  the largest systematic overestimate and the worst MAPE (27.6%) of any
  quintile — treat estimates for lower-priced properties with more caution
  than the headline MAPE (21.4%) suggests.
- **Listing attributes only.** The model has no signal on property
  condition, view, exact floor plan, legal status, or anything not captured
  by the portal's structured fields.
