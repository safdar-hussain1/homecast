# Data dictionary

Two files, same listings, different stages of the pipeline:

- **`data/gurgaon/raw/gurgaon_properties.csv`** — 3,803 listings x 23 columns,
  as collected from a public Indian property-listing portal. Untouched raw
  column names.
- **`data/gurgaon/processed/listings_clean.csv`** — 3,600 listings, output of
  `homecast clean --city gurgaon` (`src/homecast/cleaning.py`). Same columns,
  renamed to snake_case, with categorical labels applied, unpriced rows
  dropped, and percentile outliers on `price_per_sqft`/`area` trimmed. See
  the README's Methodology section for the row-by-row cleaning trail.

## Where this data came from, and what you may do with it

The snapshot is a scrape of residential listings from a public Indian
property-listing portal, redistributed in this repository so that every
published number can be reproduced end to end. **Nothing in the file or in
this repository records which portal it was taken from, or when** — no source
URL, no collection date, no metadata block — so no specific site is named
here. Naming one on the strength of the column layout alone would be a guess
dressed up as a fact.

The repository's `LICENSE` (MIT) covers the **code only**. The CSV is
third-party listing content and carries no licence grant from this project.
It is included for study and reproduction; check the originating portal's
terms before putting it to any other use.

Two things to read every number in this file, and every number derived from
it, with:

- **All prices are asking prices, not transactions.** Nobody is confirmed to
  have paid these amounts — they are what a seller listed the property for.
- **This is a single, undated snapshot.** There is no listing-date column.
  The file was first committed to this repository on **2025-08-23**; it
  reflects that market, not today's. Government registration/stamp-duty data
  would fix both of these — dated, transaction-level records — and is the
  intended upgrade path.

Units: `price` is in **₹ crore** (1 crore = ₹10,000,000), `price_per_sqft` is
in **₹**, and every area column is in **sq. ft.**

## Columns

| Raw column | Processed column | Type | Description |
|---|---|---|---|
| `property_type` | `property_type` | categorical | `flat` or `house` (independent house / builder floor) |
| `society` | `society` | categorical (676 values) | Housing society / project name |
| `sector` | `sector` | categorical (104 values) | Gurgaon sector of the listing |
| `price` | `price` | float | Asking price, ₹ crore |
| `price_per_sqft` | `price_per_sqft` | float | Asking price per sq. ft., ₹ |
| `area` | `area` | float | Headline area, sq. ft. (consistent with `price / price_per_sqft`) |
| `areaWithType` | `area_with_type` | string | Free-text area breakdown as shown on the portal |
| `bedRoom` | `bedrooms` | int | Number of bedrooms |
| `bathroom` | `bathrooms` | int | Number of bathrooms |
| `balcony` | `balcony` | ordered categorical | Number of balconies: `0`, `1`, `2`, `3`, `3+` (processed: ordered `pd.Categorical`) |
| `floorNum` | `floor_num` | float | Floor the unit is on (0 = ground) |
| `facing` | `facing` | categorical | Direction the property faces (~29% missing) |
| `agePossession` | `age_possession` | categorical | `Under Construction`, `New Property`, `Relatively New`, `Moderately Old`, `Old Property` (processed: raw value `Undefined` becomes missing) |
| `super_built_up_area` | `super_built_up_area` | float | Super built-up area, sq. ft. (~50% missing) |
| `built_up_area` | `built_up_area` | float | Built-up area, sq. ft. (~54% missing) |
| `carpet_area` | `carpet_area` | float | Carpet area, sq. ft. (~49% missing) |
| `study room` | `has_study_room` | binary | 1 if the listing includes a study room |
| `servant room` | `has_servant_room` | binary | 1 if the listing includes a servant room |
| `store room` | `has_store_room` | binary | 1 if the listing includes a store room |
| `pooja room` | `has_pooja_room` | binary | 1 if the listing includes a pooja room |
| `others` | `has_other_rooms` | binary | 1 if other extra rooms are listed |
| `furnishing_type` | `furnishing_type` | categorical | Raw: codes `0`/`1`/`2`. Processed: labelled `unfurnished` / `semi-furnished` / `furnished` |
| `luxury_score` | `luxury_score` | int (0-174) | Portal-derived score counting luxury amenities |

Every column present in the raw file is carried through to
`listings_clean.csv` under its processed name — cleaning only renames,
labels, and removes rows; it does not drop or add columns. The model's own
feature set (a subset of these columns, plus a derived sector encoding) is
documented separately in `reports/MODEL_CARD.md`.

## Known quirks handled by the pipeline

| Issue | Extent | Treatment |
|---|---|---|
| Exact duplicate listings | 126 rows | dropped |
| Missing price | 17 rows | dropped (unusable for price analysis) |
| Unit-error outliers (e.g. a listing with an implausible sq.ft./₹ figure) | 60 rows | trimmed outside the 0.5th-99.5th percentile of `price_per_sqft` and `area` |
| Missing area breakdowns (`super_built_up_area`, `built_up_area`, `carpet_area`) | roughly half of rows each | kept as optional attributes (portal-side optional fields), not imputed |
| `agePossession = "Undefined"` | 333 rows | treated as missing |

## Area basis (read before joining any other city's data to this one)

Gurgaon-area listings on major portals are conventionally quoted on a
**super-built-up** basis, and this file does not record a distinct basis
column for that reason. Other Indian metros are not consistent with this —
Mumbai listings, for instance, commonly quote **carpet area** instead — and
carpet vs. super-built-up are **not interconvertible** without knowing the
building's loading factor. `homecast ingest` (`src/homecast/ingest.py`)
requires every new city's ingestion config to declare its `area_basis`
explicitly and stamps it onto every ingested row for exactly this reason:
never assume, and never silently compare, `price_per_sqft` across cities
without checking this first.
