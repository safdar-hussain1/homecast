# Model card — HomeCast Gurgaon valuation model

## Overview

Gradient-boosted regression tree ensemble that estimates a residential
property's asking price in Gurgaon, India, from a handful of listing
attributes. Trained and evaluated on a single snapshot of scraped listing
data — see [Limitations](#limitations).

## Data provenance and snapshot nature

- Source: `data/gurgaon/raw/gurgaon_properties.csv`, 3,803 residential listings
  (flats and independent houses / builder floors) scraped from a property
  portal at one point in time. There is no listing date column and no repeat
  observations of the same property over time — this is a cross-sectional
  snapshot, not a time series.
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
0.195, `bathrooms` 0.157, `is_house` 0.085 (remaining features share the
rest). Area and the sector encoding together account for over 70% of the
model's decisions.

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
log-residuals (`log(pred) - log(actual)`) across all 5 folds. Converted to a
multiplicative range around a point estimate: **-26.9% to +37.7%**
(`exp(-0.31276) - 1` and `exp(0.31979) - 1`). For a typical prediction, the
true price is expected to fall within this range around the point estimate
about 80% of the time. The band is asymmetric: the model's overestimates run
larger in relative terms than its underestimates, consistent with the
quintile error structure below.

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
- **Area extrapolation is guarded, not extrapolated.** `homecast predict`
  rejects an `area` outside the range seen in training (`fitted.ranges`,
  from the min/max of the cleaned dataset) rather than silently
  extrapolating a tree-based model past its training distribution, where
  gradient-boosted trees produce flat, unreliable predictions.
- **Weakest at the low end.** Q1 (the cheapest fifth of listings) has both
  the largest systematic overestimate and the worst MAPE (27.6%) of any
  quintile — treat estimates for lower-priced properties with more caution
  than the headline MAPE (21.4%) suggests.
- **Listing attributes only.** The model has no signal on property
  condition, view, exact floor plan, legal status, or anything not captured
  by the portal's structured fields.
