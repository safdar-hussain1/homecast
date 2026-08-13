import numpy as np
import pandas as pd
import pytest

from homecast.features import (ALL_FEATURE_COLUMNS, FEATURE_COLUMNS,
                               MISSING_CODE, SOCIETY_SMOOTHING_M,
                               build_features, feature_columns_for,
                               fit_encoders, target)


def test_feature_matrix_shape_and_columns(clean_fixture):
    enc = fit_encoders(clean_fixture)
    X = build_features(clean_fixture, enc)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(clean_fixture)
    assert X.notna().all().all()

def test_sector_encoding_is_train_only(clean_fixture):
    train = clean_fixture[clean_fixture["sector"] != "sector 3"]
    enc = fit_encoders(train)
    assert "sector 3" not in enc.sector_ppsf              # never saw it
    X = build_features(clean_fixture, enc)                # applied to full data
    unseen = clean_fixture["sector"] == "sector 3"
    assert (X.loc[unseen, "sector_ppsf"] == enc.sector_ppsf["__global__"]).all()

def test_sector_encoding_values(clean_fixture):
    enc = fit_encoders(clean_fixture)
    s1 = clean_fixture[clean_fixture["sector"] == "sector 1"]
    assert enc.sector_ppsf["sector 1"] == pytest.approx(s1["price_per_sqft"].median())

def test_target_is_log_price(clean_fixture):
    assert np.allclose(target(clean_fixture), np.log(clean_fixture["price"]))

def test_unknown_furnishing_fails_fast(clean_fixture):
    bad = clean_fixture.copy()
    bad.loc[bad.index[0], "furnishing_type"] = "gold-plated"
    with pytest.raises(ValueError, match="furnishing"):
        build_features(bad, fit_encoders(clean_fixture))

def test_missing_age_coded_minus_one(clean_fixture):
    enc = fit_encoders(clean_fixture)
    X = build_features(clean_fixture, enc)
    missing = clean_fixture["age_possession"].isna()
    assert missing.any(), "fixture has no missing ages — the test would self-void"
    assert (X.loc[missing, "age_code"] == -1).all()

def test_sector_encoding_uses_only_given_rows(clean_fixture):
    """Dropping some of a sector's rows must change its encoding —
    proof the median comes from exactly the rows passed in."""
    s1 = clean_fixture[clean_fixture["sector"] == "sector 1"]
    # drop the highest-ppsf half of sector 1's rows
    drop_idx = s1.sort_values("price_per_sqft").index[len(s1) // 2:]
    subset = clean_fixture.drop(index=drop_idx)
    enc_full = fit_encoders(clean_fixture)
    enc_fold = fit_encoders(subset)
    assert enc_fold.sector_ppsf["sector 1"] != enc_full.sector_ppsf["sector 1"]
    assert enc_fold.sector_ppsf["sector 1"] == pytest.approx(
        subset[subset["sector"] == "sector 1"]["price_per_sqft"].median())


# ── fold-local sector statistics (mean / std / count) ─────────────────────

def test_sector_stats_values(clean_fixture):
    enc = fit_encoders(clean_fixture)
    s1 = clean_fixture[clean_fixture["sector"] == "sector 1"]
    assert enc.sector_ppsf_mean["sector 1"] == pytest.approx(s1["price_per_sqft"].mean())
    assert enc.sector_ppsf_std["sector 1"] == pytest.approx(s1["price_per_sqft"].std())
    assert enc.sector_count["sector 1"] == pytest.approx(len(s1))

def test_sector_stats_use_only_given_rows(clean_fixture):
    """Same fold-safety proof as the median encoding, for mean/std/count."""
    s1 = clean_fixture[clean_fixture["sector"] == "sector 1"]
    drop_idx = s1.sort_values("price_per_sqft").index[len(s1) // 2:]
    subset = clean_fixture.drop(index=drop_idx)
    enc_full = fit_encoders(clean_fixture)
    enc_fold = fit_encoders(subset)
    assert enc_fold.sector_ppsf_mean["sector 1"] != enc_full.sector_ppsf_mean["sector 1"]
    sub_s1 = subset[subset["sector"] == "sector 1"]
    assert enc_fold.sector_ppsf_mean["sector 1"] == pytest.approx(sub_s1["price_per_sqft"].mean())
    assert enc_fold.sector_count["sector 1"] == pytest.approx(len(sub_s1))

def test_unseen_sector_falls_back_to_global_stats(clean_fixture):
    train = clean_fixture[clean_fixture["sector"] != "sector 3"]
    enc = fit_encoders(train)
    X = build_features(clean_fixture, enc)
    unseen = clean_fixture["sector"] == "sector 3"
    assert (X.loc[unseen, "sector_ppsf_mean"] == enc.sector_ppsf_mean["__global__"]).all()
    assert (X.loc[unseen, "sector_ppsf_std"] == enc.sector_ppsf_std["__global__"]).all()
    # An unseen sector has zero TRAINING rows -- fall back to 0, not an
    # average count.
    assert (X.loc[unseen, "sector_count"] == 0).all()

def test_single_listing_sector_std_falls_back_to_global(clean_fixture):
    """std() of a single-row group is NaN; that must not leak a NaN feature."""
    one_row = clean_fixture.iloc[[0]].copy()
    one_row["sector"] = "sector solo"
    train = pd.concat([clean_fixture, one_row], ignore_index=True)
    enc = fit_encoders(train)
    assert enc.sector_ppsf_std["sector solo"] == pytest.approx(enc.sector_ppsf_std["__global__"])


# ── society smoothed target encoding ───────────────────────────────────────

def _society_frame() -> pd.DataFrame:
    """A frame with one well-sampled society (n=20), one lightly-sampled
    society (n=1), several singleton societies, and one row with a missing
    society name -- enough to exercise smoothing, fold-safety and the
    missing/unseen fallback."""
    rng = np.random.default_rng(3)
    n = 30
    ppsf = rng.uniform(4000.0, 20000.0, n).round()
    area = rng.uniform(800.0, 2000.0, n).round()
    society = ["big society"] * 20 + ["tiny society"] + [f"solo {i}" for i in range(9)]
    df = pd.DataFrame({
        "sector": "sector 1",
        "society": society,
        "property_type": "flat",
        "price": area * ppsf / 1e7,
        "price_per_sqft": ppsf,
        "area": area,
        "bedrooms": 3,
        "bathrooms": 2,
        "balcony": "2",
        "furnishing_type": "semi-furnished",
        "luxury_score": 50,
        "age_possession": "New Property",
    })
    df.loc[0, "society"] = np.nan            # a listing with no society name
    return df

def test_society_smoothing_formula():
    df = _society_frame()
    enc = fit_encoders(df)
    global_median = float(df["price_per_sqft"].median())
    big = df[df["society"] == "big society"]
    n, med = len(big), big["price_per_sqft"].median()
    expected = (n * med + SOCIETY_SMOOTHING_M * global_median) / (n + SOCIETY_SMOOTHING_M)
    assert enc.society_ppsf["big society"] == pytest.approx(expected)

def test_society_smoothing_shrinks_singleton_toward_global_median():
    """A 1-listing society's smoothed value must sit strictly between its own
    raw price_per_sqft and the global median -- proof m=10 actually shrinks."""
    df = _society_frame()
    enc = fit_encoders(df)
    global_median = float(df["price_per_sqft"].median())
    tiny = df[df["society"] == "tiny society"]
    raw = float(tiny["price_per_sqft"].iloc[0])
    smoothed = enc.society_ppsf["tiny society"]
    assert raw != pytest.approx(global_median), "fixture self-voids: raw == global by chance"
    lo, hi = sorted([raw, global_median])
    assert lo < smoothed < hi

def test_society_encoding_uses_only_given_rows():
    """Fold-safety for society, same style as the sector proof: dropping some
    of a society's rows must change its smoothed encoding."""
    df = _society_frame()
    big = df[df["society"] == "big society"]
    drop_idx = big.sort_values("price_per_sqft").index[len(big) // 2:]
    subset = df.drop(index=drop_idx)
    enc_full = fit_encoders(df)
    enc_fold = fit_encoders(subset)
    assert enc_fold.society_ppsf["big society"] != enc_full.society_ppsf["big society"]
    sub_big = subset[subset["society"] == "big society"]
    global_median = float(subset["price_per_sqft"].median())
    n, med = len(sub_big), sub_big["price_per_sqft"].median()
    expected = (n * med + SOCIETY_SMOOTHING_M * global_median) / (n + SOCIETY_SMOOTHING_M)
    assert enc_fold.society_ppsf["big society"] == pytest.approx(expected)

def test_society_encoding_only_sees_training_rows(monkeypatch):
    """Structural mirror of the sector fold-loop test: fit_encoders must be
    handed the training fold only, never the full frame, so its society map
    can never see a held-out row's own price."""
    import homecast.model as model_mod
    df = _society_frame()
    seen: list[int] = []
    real = model_mod.fit_encoders

    def spy(frame):
        seen.append(len(frame))
        return real(frame)

    monkeypatch.setattr(model_mod, "fit_encoders", spy)
    model_mod.evaluate(df, n_splits=3)
    assert seen, "fit_encoders was never called — the spy is not wired up"
    assert len(df) not in seen, (
        f"fit_encoders was handed all {len(df)} rows: the fold loop is leaking")

def test_unseen_society_falls_back_to_global_median():
    """In `_society_frame` every row is "sector 1" -- the whole dataset is one
    sector, so its sector rate and the global median are numerically the same
    value here. See the dedicated fallback-chain tests below for a fixture
    with >1 sector, which pins down that the fallback is really "the row's
    OWN sector rate", not the global median directly."""
    df = _society_frame()
    train = df[df["society"] != "big society"]
    enc = fit_encoders(train)
    assert "big society" not in enc.society_ppsf
    X = build_features(df, enc)
    unseen = df["society"] == "big society"
    assert (X.loc[unseen, "society_ppsf"] == enc.society_ppsf["__global__"]).all()

def test_missing_society_falls_back_to_global_median():
    """A listing with no society name at all gets the row's sector rate (here
    numerically equal to the global median -- see the note above); it is not
    bucketed as its own pseudo-society."""
    df = _society_frame()
    enc = fit_encoders(df)
    X = build_features(df, enc)
    missing = df["society"].isna()
    assert missing.any(), "fixture has no missing society — the test would self-void"
    assert (X.loc[missing, "society_ppsf"] == enc.society_ppsf["__global__"]).all()

def test_known_society_uses_its_own_smoothed_rate():
    """A recognised society must use its own encoding, not fall back at all."""
    df = _society_frame()
    enc = fit_encoders(df)
    X = build_features(df, enc)
    known = df["society"] == "big society"
    assert np.allclose(X.loc[known, "society_ppsf"], enc.society_ppsf["big society"])


# ── society_ppsf fallback chain: society -> that row's sector -> global ────
# (needs >1 sector so "the row's own sector rate" is distinguishable from
# "the city-wide global median" -- `_society_frame` is a single sector, where
# the two happen to coincide.)

def _two_sector_frame() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 60
    sector = ["sector hi"] * 30 + ["sector lo"] * 30
    ppsf = np.array([20000.0] * 30 + [6000.0] * 30) * np.exp(rng.normal(0.0, 0.01, n))
    area = rng.uniform(800.0, 2000.0, n)
    society = [f"soc {i}" for i in range(n)]   # every society a singleton
    return pd.DataFrame({
        "sector": sector, "society": society, "property_type": "flat",
        "price": area * ppsf / 1e7, "price_per_sqft": ppsf, "area": area,
        "bedrooms": 3, "bathrooms": 2, "balcony": "2",
        "furnishing_type": "semi-furnished", "luxury_score": 50,
        "age_possession": "New Property",
    })

def _query_row(**over) -> pd.DataFrame:
    base = dict(sector="sector hi", society="unseen society", property_type="flat",
                price_per_sqft=np.nan, area=1000.0, bedrooms=3, bathrooms=2,
                balcony="2", furnishing_type="semi-furnished", luxury_score=50,
                age_possession="New Property")
    base.update(over)
    return pd.DataFrame([base])

def test_unknown_society_with_known_sector_falls_back_to_that_sector_rate():
    df = _two_sector_frame()
    enc = fit_encoders(df)
    global_median = enc.society_ppsf["__global__"]
    sector_hi_rate = enc.sector_ppsf["sector hi"]
    assert sector_hi_rate != pytest.approx(global_median), \
        "fixture self-voids: sector hi's rate coincides with the global median"
    q = _query_row(sector="sector hi", society="a brand new building nobody trained on")
    assert "a brand new building nobody trained on" not in enc.society_ppsf
    X = build_features(q, enc)
    assert X["society_ppsf"].iloc[0] == pytest.approx(sector_hi_rate)
    assert X["society_ppsf"].iloc[0] != pytest.approx(global_median)

def test_unknown_society_and_unknown_sector_falls_back_to_global_median():
    df = _two_sector_frame()
    enc = fit_encoders(df)
    q = _query_row(sector="a sector nobody trained on", society="a society nobody trained on")
    X = build_features(q, enc)
    assert X["sector_ppsf"].iloc[0] == pytest.approx(enc.sector_ppsf["__global__"])
    assert X["society_ppsf"].iloc[0] == pytest.approx(enc.society_ppsf["__global__"])

def test_known_society_is_not_overridden_by_the_sector_fallback():
    df = _two_sector_frame()
    enc = fit_encoders(df)
    known_society = df["society"].iloc[0]
    known_sector = df["sector"].iloc[0]
    q = _query_row(sector=known_sector, society=known_society)
    X = build_features(q, enc)
    assert X["society_ppsf"].iloc[0] == pytest.approx(enc.society_ppsf[known_society])


# ── balcony ─────────────────────────────────────────────────────────────

def test_balcony_code_mapping(clean_fixture):
    enc = fit_encoders(clean_fixture)
    X = build_features(clean_fixture, enc)
    want = {"0": 0, "1": 1, "2": 2, "3": 3, "3+": 4}
    for label, code in want.items():
        rows = clean_fixture["balcony"].astype(str) == label
        if rows.any():
            assert (X.loc[rows, "balcony_code"] == code).all()

def test_missing_balcony_coded_minus_one(clean_fixture):
    bad = clean_fixture.copy()
    bad["balcony"] = bad["balcony"].astype(object)
    bad.loc[bad.index[0], "balcony"] = None
    enc = fit_encoders(bad)
    X = build_features(bad, enc)
    assert X.loc[bad.index[0], "balcony_code"] == -1


# --- Heterogeneous cities: the feature set degrades per city ---------------
#
# The public Gurgaon feed carries furnishing, an amenity score, possession
# age, balconies and a building name. A third-party feed for another metro
# routinely carries none of those, and sometimes carries things Gurgaon's
# does not. These tests pin the rule that decides what each city gets, and
# -- more importantly -- pin that train and predict can never disagree about
# it, which is the failure that would silently mispredict rather than crash.

def _reduced_frame(n: int = 60, *, society: bool = False,
                   amenity: bool = False) -> pd.DataFrame:
    """A frame shaped like a sparse third-party feed: price, area, bedrooms
    and a locality, and nothing else unless asked for."""
    rng = np.random.default_rng(3)
    area = rng.uniform(500, 2500, n).round()
    ppsf = rng.uniform(4000, 15000, n).round()
    df = pd.DataFrame({
        "sector": rng.choice(["andheri", "bandra", "powai"], n),
        "area": area,
        "bedrooms": rng.integers(1, 5, n),
        "price": (area * ppsf / 1e7),
        "price_per_sqft": ppsf,
    })
    if society:
        df["society"] = rng.choice(["tower a", "tower b"], n)
    if amenity:
        df["amenity_count"] = rng.integers(0, 20, n).astype(float)
        df["is_resale"] = rng.integers(0, 2, n).astype(float)
    return df


def test_full_frame_selects_exactly_the_public_feature_set(clean_fixture):
    """The catalogue must not quietly change what Gurgaon trains on. If this
    fails, the public model's feature vector moved and its published metrics
    no longer describe the shipped model."""
    assert feature_columns_for(clean_fixture) == FEATURE_COLUMNS


def test_sparse_frame_drops_every_feature_it_cannot_supply():
    cols = feature_columns_for(_reduced_frame())
    assert cols == ["area", "bedrooms", "sector_ppsf", "sector_ppsf_mean",
                    "sector_ppsf_std", "sector_count"]
    for absent in ("society_ppsf", "furnishing_code", "luxury_score",
                   "age_code", "balcony_code", "is_house", "bathrooms"):
        assert absent not in cols


def test_a_city_with_extra_columns_gains_those_features():
    cols = feature_columns_for(_reduced_frame(amenity=True))
    assert "amenity_count" in cols and "is_resale" in cols


def test_selected_features_always_follow_catalogue_order():
    """Two cities that share a feature must agree on where it sits relative
    to the others, or a feature vector built for one would be silently
    misread as the other's."""
    for df in (_reduced_frame(society=True, amenity=True), _reduced_frame()):
        cols = feature_columns_for(df)
        assert cols == [c for c in ALL_FEATURE_COLUMNS if c in set(cols)]


def test_frame_without_a_core_feature_is_refused():
    df = _reduced_frame().drop(columns=["bedrooms"])
    with pytest.raises(ValueError, match="bedrooms"):
        feature_columns_for(df)


def test_fit_encoders_survives_a_city_with_no_society_column():
    """A feed with no building names must not crash the encoder fit; it gets
    a society map holding only the global fallback, which is never consulted
    because society_ppsf is not in its feature list."""
    enc = fit_encoders(_reduced_frame())
    assert set(enc.society_ppsf) == {"__global__"}


def test_build_features_never_touches_an_absent_optional_column():
    df = _reduced_frame()
    X = build_features(df, fit_encoders(df))
    assert "society_ppsf" not in X.columns
    assert X.notna().all().all()


def test_build_features_respects_an_explicit_column_list():
    df = _reduced_frame(society=True)
    enc = fit_encoders(df)
    subset = ["area", "sector_ppsf"]
    X = build_features(df, enc, subset)
    assert list(X.columns) == subset


def test_requesting_a_feature_the_frame_cannot_supply_fails_loudly():
    """The dangerous version of this bug is silent: a model fit with
    society_ppsf, later handed a frame without a society column, must refuse
    rather than quietly predict on a different feature set."""
    df = _reduced_frame()
    with pytest.raises(ValueError, match="society"):
        build_features(df, fit_encoders(df), ["area", "society_ppsf"])


def test_requesting_a_non_catalogue_feature_fails_loudly():
    df = _reduced_frame()
    with pytest.raises(ValueError, match="catalogue"):
        build_features(df, fit_encoders(df), ["area", "vibes"])


def test_unrecorded_amenity_count_becomes_the_missing_code_not_zero():
    """These feeds mark 'amenities were never captured for this listing' with
    a blank, and a blank must stay distinguishable from a genuine zero
    amenities. Collapsing the two is how an amenity feature ends up learning
    which listings were poorly recorded instead of which are well appointed."""
    df = _reduced_frame(amenity=True)
    df.loc[df.index[:5], "amenity_count"] = np.nan
    X = build_features(df, fit_encoders(df))
    assert (X["amenity_count"].iloc[:5] == MISSING_CODE).all()
    assert (X["amenity_count"].iloc[5:] >= 0).all()


def test_explicit_columns_reproduce_the_inferred_result_on_a_full_frame(clean_fixture):
    enc = fit_encoders(clean_fixture)
    assert build_features(clean_fixture, enc).equals(
        build_features(clean_fixture, enc, FEATURE_COLUMNS))
