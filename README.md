# HomeCast

Residential property price intelligence for Indian cities. A gradient-boosted
valuation model trained on real listing data, shipped as an installable
package with a CLI, plus a static dashboard where the same model runs
entirely in the visitor's browser.

**[Live dashboard](https://safdar-hussain1.github.io/homecast/)** — Gurgaon is live today; the city registry is built so more cities plug in.

![HomeCast dashboard](reports/figures/dashboard.png)

## What it does

- **Estimator** — `homecast predict` takes a sector, property type, bedrooms,
  bathrooms, area, furnishing, and luxury score, and returns a point estimate
  in ₹ crore plus an 80% uncertainty band.
- **CLI** — one tool covers the full pipeline: `clean` (raw listings →
  cleaned dataset), `train` (fit the model, write metrics + a dashboard
  export), `evaluate` (cross-validated metrics only, no artifacts written),
  `predict` (price a single property), and `export-dashboard` (regenerate the
  browser-side model file from an already-trained model).
- **Per-city pipeline** — every city is a `City` entry in a small registry
  (`src/homecast/cities.py`) pointing at its raw CSV, cleaned CSV, and model
  directory, plus a cleaning function registered in `PIPELINES`. Gurgaon is
  the first city; adding another is described below.

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/safdar-hussain1/homecast.git
cd homecast

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

homecast train --city gurgaon
```

Output:

```
                           MAE (lakh)   MAPE %     R2
HomeCast model                   54.4     21.4  0.806
Sector-median baseline           72.8     26.6  0.653
Global-median baseline          114.9     38.1  0.299
Artifacts -> .../models/gurgaon
```

Then price a property:

```bash
homecast predict --city gurgaon --sector "sector 65" --type flat \
  --bhk 3 --bath 3 --area 1800 --furnishing semi-furnished --luxury 60
```

```
Estimated price: Rs 2.78 Cr (range 2.03 - 3.83 Cr)
```

Note the `--city` flag comes **after** the subcommand (`homecast train --city
gurgaon`, not `homecast --city gurgaon train`) — each subcommand owns its own
`--city` argument, so a top-level one is rejected by design.

`data/gurgaon/raw/gurgaon_properties.csv` and the cleaned/trained artifacts
under `models/gurgaon/` are committed, so `train` reproduces the numbers
above without needing `clean` first (it runs `clean` automatically if the
cleaned CSV is missing). Run `homecast clean --city gurgaon` directly to see
the row-by-row cleaning log.

## Results

5-fold cross-validation, out-of-fold predictions, n = 3,600:

| Model                    | MAE (lakh) | MAPE (%) | R²    |
|---------------------------|-----------:|---------:|------:|
| HomeCast model (GBR)      |      54.4  |    21.4  | 0.806 |
| Sector-median ₹/sqft rule |      72.8  |    26.6  | 0.653 |
| Global-median ₹/sqft rule |     114.9  |    38.1  | 0.299 |

The sector-median ₹/sqft rule is the naive baseline an agent uses without a
model: median ₹/sqft for the listing's sector, times the property's area.
The model cuts MAE by roughly a quarter against that baseline.

**The band**: each estimate ships with a range, not just a point. It's the
10th–90th percentile of out-of-fold log-residuals, converted to a
multiplicative range: roughly **-27% to +38%** around the point estimate.
The band is asymmetric — the model's misses skew larger on the upside than
the downside, in relative terms — see the error structure in Methodology.

Model parameters: `n_estimators=300, max_depth=3, learning_rate=0.06,
subsample=0.9, random_state=7`. These are HomeCast's own default parameters
(not scikit-learn's), chosen up front and never tuned against the results —
they beat both baselines on the first run, so the planned learning-rate x
max-depth grid was never needed.

Full detail: [`reports/MODEL_CARD.md`](reports/MODEL_CARD.md).

## Methodology

**Cleaning trail** (`homecast clean --city gurgaon`, logged step by step):

| Step | Rows removed | Rows remaining |
|---|---:|---:|
| Raw feed | — | 3,803 |
| Drop exact duplicate listings | 126 | 3,677 |
| Drop listings without a price | 17 | 3,660 |
| Trim percentile outliers (`price_per_sqft`, `area`; 0.5th–99.5th pct) | 60 | 3,600 |

**Features**: `area`, `bedrooms`, `bathrooms`, `is_house`, `furnishing_code`,
`luxury_score`, `age_code`, `sector_ppsf`. The last one — the sector encoded
as its median ₹/sqft — is by far the most informative categorical signal,
but it's also the one most prone to leaking the target if handled carelessly.

**Fold-safe encoding**: `sector_ppsf` is *re-learned inside every training
fold* of the 5-fold cross-validation, from that fold's training rows only,
then applied to the held-out fold. Encoding the sector on the full dataset
before splitting would leak information about a listing's own price (and its
neighbors' prices) into the feature used to predict it — cross-validated
error would look better than what the model achieves on genuinely new
listings. Every fold re-fits its own encoding for exactly this reason.

**CV design**: 5-fold `KFold(shuffle=True, random_state=7)` over all 3,600
listings; predictions are collected out-of-fold, so every metric above is
computed on rows the model never trained on for that prediction. Feature
importances from the trained model: `area` 0.537, `sector_ppsf` 0.195,
`bathrooms` 0.157, `is_house` 0.085 — area and sector together account for
over 70% of the model's decisions.

**Error structure by price quintile** (mean signed log-residual, positive =
overestimate): Q1 +0.129, Q2 +0.033, Q3 −0.001, Q4 −0.033, Q5 −0.129 —
monotonic. The model **overestimates the cheapest listings and
underestimates the most expensive ones**; Q1 also has the worst MAPE of the
five quintiles, at 27.6%.

## Market findings

From the exploratory analysis (`notebooks/gurgaon_real_estate_eda.ipynb`),
on the cleaned 3,600-listing dataset (2,799 flats, 801 houses):

1. **Two markets in one city** — houses run about 3× the price of flats
   (median flat ₹1.39 Cr vs median house ₹4.25 Cr; citywide median ₹1.54 Cr,
   mean ₹2.51 Cr, max ₹31.5 Cr).
2. **Location is a step function** — sector median prices span roughly 30×
   across the city.
3. **3 BHK dominates supply** — it's the single most common bedroom count by
   a wide margin.
4. **Bathrooms ≈ bedrooms** — the two are nearly interchangeable size
   signals (r ≈ 0.91).
5. **A Simpson's paradox in property age** — in aggregate, old listings look
   like the priciest per sq.ft., but that's only because 58% of old listings
   are houses. Within flats alone, under-construction stock is the priciest
   per sq.ft., not old stock.

## Repo structure

```
data/
  gurgaon/raw/gurgaon_properties.csv        # raw scraped listings (3,803 rows)
  gurgaon/processed/listings_clean.csv      # output of `homecast clean` (3,600 rows)
  DATA_DICTIONARY.md                        # column reference for both files
src/homecast/
  cities.py       # city registry: City dataclass + CITIES dict
  cleaning.py     # cleaning pipeline + PIPELINES registry
  features.py     # feature engineering, fold-safe sector encoding
  model.py        # GradientBoostingRegressor training + cross-validated evaluate()
  valuation.py    # Query -> price estimate, with the uncertainty band
  export.py       # write the browser-side model.json export
  cli.py          # `homecast` command line: clean, train, evaluate, predict, export-dashboard
  plotting.py     # shared plot styling for the notebooks
tests/            # 44 tests covering cities, cleaning, features, model, valuation, export, CLI
notebooks/
  gurgaon_real_estate_eda.ipynb   # exploratory analysis, market findings
  valuation_model.ipynb           # model development, CV, error analysis
scripts/
  build_dashboard.py       # renders docs/index.html from dashboard_template.html + model.json
  dashboard_template.html  # dashboard page template (model runs in-browser)
models/gurgaon/
  model.json      # browser-portable export (tree structure, band, sector medians)
  metrics.json    # CV metrics vs both baselines
  model.joblib    # trained sklearn model (gitignored; regenerate with `homecast train`)
docs/index.html    # the built dashboard, served by GitHub Pages
reports/figures/    # exported charts (EDA + model diagnostics)
reports/MODEL_CARD.md
pyproject.toml, requirements.txt, LICENSE
```

## Adding a city

The registry is deliberately small so a new city is a couple of additions,
not a rewrite:

1. Add a `City` entry in `src/homecast/cities.py`'s `CITIES` dict — key,
   display name, and the raw CSV filename (paths for `processed` and
   `models` follow the standard `data/<key>/...` / `models/<key>/`
   convention automatically).
2. Write a cleaning function for that city's raw feed and register it in
   `PIPELINES` in `src/homecast/cleaning.py` — it takes the raw DataFrame and
   returns `(cleaned_df, log_lines)`, same contract as Gurgaon's
   `clean_raw_data`.
3. Run `homecast train --city <key>`. Training, evaluation, feature
   engineering, the CLI, and the dashboard export all key off the registry —
   nothing else needs to change.

## Tech stack

Python, pandas, NumPy, scikit-learn, joblib, Matplotlib, Jupyter. The
dashboard's in-browser predictions are plain JavaScript that walks the
exported gradient-boosted trees directly — no ML runtime in the browser. It
matches the Python model to about 1e-14 (tested at 1e-9 tolerance).

## License

[MIT](LICENSE)
