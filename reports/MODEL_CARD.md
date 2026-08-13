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
  The file was first committed to this repository on **2025-08-23**; every
  number in this card reflects that market, not today's, and there is
  currently no mechanism to detect or correct for drift since then. See
  [Limitations](#limitations).
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
| `sector_ppsf` | Sector target-encoded as median ₹/sqft |
| `sector_ppsf_mean` | Sector's mean ₹/sqft (fold-local; unseen sector → global mean) |
| `sector_ppsf_std` | Sector's ₹/sqft standard deviation (fold-local; unseen sector → global std) |
| `sector_count` | Number of training-fold listings in the sector (unseen sector → 0) |
| `society_ppsf` | Society (housing project) target-encoded, smoothed toward the global median — see [Society masking](#society-masking-and-graceful-degradation) |
| `balcony_code` | `0`=0, `1`=1, `2`=2, `3`=3, `3+`=4, missing=-1 |

Feature importances from the final trained model (society-given
configuration, full precision in `models/gurgaon/model.json` →
`feature_importances`): `area` 0.5756, `society_ppsf` 0.1481, `bathrooms`
0.0800, `sector_ppsf` 0.0549, `is_house` 0.0477, `sector_ppsf_mean` 0.0350,
`luxury_score` 0.0137, `bedrooms` 0.0121, `balcony_code` 0.0098,
`sector_count` 0.0085, `sector_ppsf_std` 0.0083, `age_code` 0.0049,
`furnishing_code` 0.0016. `area` and `society_ppsf` alone account for over
72% of the model's decisions — see [Society masking](#society-masking-and-graceful-degradation)
for why `society_ppsf` being this dominant required a deliberate design fix.
These come from `GradientBoostingRegressor.feature_importances_` and are
exported verbatim to `models/gurgaon/model.json` under
`feature_importances`; the dashboard reads them from the payload, and
`tests/test_export.py` asserts the exported values equal the fitted model's.

## Fold-safe encoding

Every encoding above that is derived from the training data (`sector_ppsf`,
`sector_ppsf_mean`, `sector_ppsf_std`, `sector_count`, `society_ppsf`) is
**re-learned from scratch inside every cross-validation training fold**,
using only that fold's training rows, then applied to the held-out fold.
Computing any of them on the full dataset before splitting into train/test
would leak each listing's own price (and its sector-/society-mates' prices)
into a feature used to predict that same price, inflating the apparent
accuracy. This is implemented in `src/homecast/features.py:fit_encoders` (one
call per fold, producing a frozen `Encoders` object) + `build_features`, and
called once per fold in `src/homecast/model.py:evaluate`.
`society_ppsf` is a smoothed median target encoding,
`(n·society_median + 10·global_median) / (n + 10)` — the shrinkage constant
(`m=10`) is hardcoded, not tuned against the reported CV numbers, because 488
of the dataset's 665 societies have fewer than 5 listings and an untuned,
generous shrinkage keeps thin societies from overfitting to a handful of
prices.

## Society masking and graceful degradation

`society_ppsf` turned out to be the model's second-largest feature (0.148,
right behind `area`) — real, valuable signal. But the CLI and the dashboard
both treat society as optional (a user often doesn't know their building's
exact listing-portal name), and a model trained to lean this hard on a
feature it can't always get is a trap: measured directly, a model trained
*without* any mitigation scored **17.8% MAPE** in cross-validation when every
row supplied its real society at test time, but a real caller who didn't
name one — the normal case — got the plain global-median fallback and
**32.95% MAPE**, worse than the 21.4% this entire upgrade was meant to beat.

The fix, implemented in `src/homecast/model.py`
(`SOCIETY_MASK_FRACTION = 0.5`): during training, `society_ppsf` is
overwritten with the row's own `sector_ppsf` on a random ~50% of *training*
rows (seeded `np.random.default_rng(_SOCIETY_MASK_SEED)`, `_SOCIETY_MASK_SEED
= 7`, reproducible), in both `evaluate()`'s fold loop and `train_final()`.
This teaches the model not to over-rely on society. At predict time, an
unknown/omitted society now falls back to **the row's own sector rate**
(`features.py`'s fallback chain: known society → its rate; unknown/missing
→ this row's sector rate; sector also unknown → global median) rather than a
flat citywide constant, mirrored exactly in the dashboard's JS
`featureRow()`.

Result: society-known CV accuracy moved from 17.8% to **18.4%** MAPE (a small
cost — masking makes the model slightly more conservative even when it does
get a real society), while society-unknown accuracy moved from 32.95% to
**19.9%** MAPE (a large gain) — `evaluate()` now reports both configurations
(`model` and `model_no_society`) from the *same* out-of-fold predictions and
the same masked-training estimator, differing only in whether society is
withheld at predict time, and both are surfaced in `metrics.json`, the
dashboard export payload, and the CLI's printed table, so neither the
flattering nor the unflattering number is ever shown alone.

## Model parameters

`sklearn.ensemble.GradientBoostingRegressor`, with HomeCast's own project
defaults (`DEFAULT_PARAMS` in `src/homecast/model.py` — not scikit-learn's
library defaults). The Phase 2 values below were specified in the Phase 2
plan before this evaluation ran and have not been changed since based on
the CV results this card reports — they were not tuned against these numbers:

```
n_estimators   = 500
max_depth      = 5
learning_rate  = 0.05
subsample      = 0.9
random_state   = 7
```

`src/homecast/model.py` also defines a named model registry (`MODELS`): the
`"default"` GBR above, which is the only one ever exported to the browser,
and an `"accurate"` `ExtraTreesRegressor(n_estimators=100, random_state=7,
n_jobs=-1)`, CLI-only (`homecast train --model accurate` /
`homecast evaluate --model accurate`). `export.py` raises a clear error if
asked to export `"accurate"` — `ExtraTreesRegressor` has no
`init_`/`learning_rate` for the browser-side tree walker to use, and shipping
a model the JS walker can't actually run would be silently broken, not just
unsupported.

## Reproducing these numbers

Every figure in this card was produced on Python 3.12.13 with pandas 3.0.3,
NumPy 2.5.1, scikit-learn 1.9.0, SciPy 1.18.0, joblib 1.5.3 and Matplotlib
3.11.0 — the pinned set in `requirements.txt`. Install that file before
`pip install -e .`. The pipeline is fully seeded (`random_state=7` for the
model, the `KFold` split, and the society-masking RNG), so those versions
reproduce the metrics exactly (re-verified this session: a from-scratch
`homecast train --city gurgaon` run reproduced `metrics.json` bit-for-bit);
a materially different scikit-learn version may differ in the last decimal
or two.

## Cross-validation protocol

5-fold `KFold(shuffle=True, random_state=7)` over all 3,600 cleaned
listings. Predictions are collected out-of-fold — every metric below is
computed on rows the model never saw during training for that fold, for both
the society-given and society-withheld read of the same masked-training
estimator (see [Society masking](#society-masking-and-graceful-degradation)).
The final shipped model is then refit on all 3,600 rows using the same
parameters, the same society masking, and encodings learned from the full
dataset (only the CV metrics use fold-local encodings; the deployed model
uses ones learned from everything it has).

## Metrics vs. baselines

Out-of-fold, 5-fold CV, n = 3,600:

| Model | MAE (₹ lakh) | MAPE (%) | R² |
|---|---:|---:|---:|
| HomeCast model — society given | 47.6 | 18.4 | 0.825 |
| HomeCast model — society **not** given (the default a browsing user gets) | 51.7 | 19.9 | 0.811 |
| Sector-median ₹/sqft rule | 72.8 | 26.6 | 0.653 |
| Global-median ₹/sqft rule | 114.9 | 38.1 | 0.299 |

Full precision (`models/gurgaon/metrics.json`): model (society given) MAE =
47.637769... lakh, MAPE = 18.368716...%, R² = 0.825473...; model_no_society
MAE = 51.656840... lakh, MAPE = 19.914631...%, R² = 0.811163...; sector
baseline MAE = 72.786742... lakh, MAPE = 26.645038...%, R² = 0.652677...;
global baseline MAE = 114.936236... lakh, MAPE = 38.078945...%, R² =
0.299244....

Both baselines are the pricing rules an agent uses without a model: multiply
a median ₹/sqft figure (global, or specific to the listing's sector) by the
listing's area. The model beats the sector-median rule by roughly 30%
(29.0%) on MAE even in its harder, society-withheld configuration — and by
roughly a third (34.6%) with a society given — and the sector-median rule in
turn beats the global-median rule — the expected ordering holds.
**Neither row above is quoted alone anywhere in this project** — which one
applies depends on whether the caller supplied a society, and both are
carried through `metrics.json`, the dashboard payload, and the CLI's printed
table for exactly that reason.

## Uncertainty band

`models/gurgaon/model.json` → `band`: `[-0.24198382955972036,
0.26662858905450754]`, the 10th and 90th percentile of the out-of-fold
log-residuals (society-given configuration) across all 5 folds. The residual
is defined as `log(pred) - log(actual)`, so `actual = pred * exp(-residual)`
— **the sign is negated when the residual band becomes a price band.** The
90th-percentile residual (the largest overestimates) sets the *low* end of
the true price and the 10th-percentile residual sets the *high* end:

```
lo = pred * exp(-0.26663) = pred * 0.7660   ->  -23.4%
hi = pred * exp(+0.24198) = pred * 1.2738   ->  +27.4%
```

So the multiplicative range around a point estimate is **-23.4% to +27.4%**,
and the true price is expected to fall inside it about 80% of the time —
narrower than the model's pre-society-masking band, consistent with the
lower MAE/MAPE that masking produced.

The percentage range is lopsided, but the band underneath it is
near-symmetric in log space. Exponentiating a symmetric log interval always
produces a larger positive percentage than negative one
(`exp(x) - 1 > |exp(-x) - 1|` for any `x > 0`), so this shape is arithmetic
and would look the same for perfectly symmetric errors — it is not evidence
that the model's misses skew one way. Where the errors genuinely are uneven
is across the price range — see the quintile structure below.

## Error structure by price quintile

Mean signed log-residual and MAPE by price quintile (society-given CV
configuration; positive residual = model overestimates), recomputed this
session against the current masked-training model — the previously-published
figures were for the pre-Phase-2 model and are no longer accurate:

| Quintile | Mean signed log-residual | MAPE |
|---|---:|---:|
| Q1 (cheapest) | +0.110 | 24.8% |
| Q2 | +0.022 | 15.2% |
| Q3 | +0.008 | 14.1% |
| Q4 | −0.016 | 17.9% |
| Q5 (priciest) | −0.088 | 19.6% |

The signed residual is still directionally monotonic — the model
**overestimates the cheapest listings and underestimates the most
expensive ones**, the classic pull toward the middle of a log-target
regression — and Q1 remains the least accurate quintile in percentage terms.
The absolute MAPE gap between Q1 and the middle quintiles narrowed
considerably versus the pre-Phase-2 model, consistent with the overall MAPE
improvement.

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

The two most important caveats on this whole project are these first two —
everything else below is secondary to them:

- **Asking prices, not transaction prices.** The training data is listing
  prices scraped from a portal. Actual sale prices — after negotiation — are
  not observed. **The model learns what sellers ask, not what buyers pay**,
  and in the Indian residential market those routinely differ. Government
  property-registration data (stamp-duty / IGRS records) is transaction-level
  and dated, which would fix this limitation and the vintage one below at the
  same time — it is the intended upgrade path, not yet built.
- **Single, undated snapshot — not today's market.** The dataset has **no
  date column at all**. It is one cross-sectional pull, first committed to
  this repository on **2025-08-23**. Every metric, band, and prediction in
  this card reflects that market, not today's, and there is currently no
  mechanism to detect or correct for drift since then. Retrain on fresher,
  dated data (see the note above) before trusting this for current pricing.
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
  the largest systematic overestimate and the worst MAPE (24.8%) of any
  quintile — treat estimates for lower-priced properties with more caution
  than the headline MAPE (18.4% with a society given, 19.9% without)
  suggests.
- **Listing attributes only.** The model has no signal on property
  condition, view, exact floor plan, legal status, or anything not captured
  by the portal's structured fields.
