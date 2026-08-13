import numpy as np
import pandas as pd
import pytest

from homecast.features import (FEATURE_COLUMNS, SOCIETY_SMOOTHING_M,
                               build_features, fit_encoders, target)


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
    df = _society_frame()
    train = df[df["society"] != "big society"]
    enc = fit_encoders(train)
    assert "big society" not in enc.society_ppsf
    X = build_features(df, enc)
    unseen = df["society"] == "big society"
    assert (X.loc[unseen, "society_ppsf"] == enc.society_ppsf["__global__"]).all()

def test_missing_society_falls_back_to_global_median():
    """A listing with no society name at all gets the plain global median —
    it is not bucketed as its own pseudo-society."""
    df = _society_frame()
    enc = fit_encoders(df)
    X = build_features(df, enc)
    missing = df["society"].isna()
    assert missing.any(), "fixture has no missing society — the test would self-void"
    assert (X.loc[missing, "society_ppsf"] == enc.society_ppsf["__global__"]).all()


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
