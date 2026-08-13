import json

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from homecast.cities import City, get_city
from homecast.features import build_features
from homecast.model import predict_price, train_final
from homecast.export import export_city, predict_from_export, write_export


@pytest.fixture()
def payload(clean_fixture):
    fitted = train_final(clean_fixture)
    return fitted, export_city(fitted, clean_fixture, get_city("gurgaon"))

def test_parity_python_vs_export(clean_fixture, payload):
    """The exported JSON must reproduce sklearn's predictions exactly."""
    fitted, p = payload
    X = build_features(clean_fixture, fitted.encoders)
    want = predict_price(fitted, X)
    got = np.array([predict_from_export(p, row) for row in X.to_numpy()])
    assert np.allclose(got, want, rtol=0, atol=1e-9)

def test_payload_is_json_serializable(payload, tmp_path):
    _, p = payload
    out = tmp_path / "gurgaon.json"
    write_export(p, out)
    round_tripped = json.loads(out.read_text())
    assert round_tripped["feature_order"][0] == "area"
    assert len(round_tripped["model"]["trees"]) == p["model"]["trees"].__len__()

def test_feature_importances_match_the_fitted_model(payload):
    """The dashboard reads these instead of hardcoding them, so they must be
    the estimator's own numbers, aligned to feature_order."""
    fitted, p = payload
    imp = p["feature_importances"]
    assert list(imp) == p["feature_order"]
    want = dict(zip(p["feature_order"], fitted.model.feature_importances_))
    assert imp == {name: float(v) for name, v in want.items()}
    assert sum(imp.values()) == pytest.approx(1.0)

def test_no_nan_in_payload(payload, tmp_path):
    _, p = payload
    write_export(p, tmp_path / "x.json")   # allow_nan=False raises on NaN

def test_residual_hist_shape(payload):
    _, p = payload
    assert len(p["residual_hist"]["edges"]) == 21
    assert len(p["residual_hist"]["counts"]) == 20

def test_sectors_records_shape_and_filter(clean_fixture, payload):
    """Sectors aggregation: >=30-listing filter, key names, sort order."""
    fitted, _ = payload
    # "big sector": 40 rows -> survives the >=30 filter.
    # "mid sector": 35 rows with price doubled -> also survives, and its median
    # is unambiguously higher than "big sector"'s, so descending sort order is
    # deterministic regardless of which rows the samples happen to draw.
    # "small sector": 10 rows -> filtered out.
    big = clean_fixture.sample(n=40, replace=True, random_state=1).assign(sector="big sector")
    mid = clean_fixture.sample(n=35, replace=True, random_state=3).assign(sector="mid sector")
    mid = mid.assign(price=mid["price"] * 2)
    small = clean_fixture.sample(n=10, replace=True, random_state=2).assign(sector="small sector")
    df = pd.concat([big, mid, small], ignore_index=True)
    p = export_city(fitted, df, get_city("gurgaon"))
    names = [s["name"] for s in p["sectors"]]
    assert names == ["mid sector", "big sector"]
    rec = p["sectors"][1]
    assert set(rec) == {"name", "n", "median_price_cr", "median_ppsf"}
    assert rec["n"] == 40
    assert rec["median_price_cr"] == pytest.approx(big["price"].median())


# ── new fold-local encoders reach the payload ──────────────────────────────

def test_encodings_carry_every_fold_local_map(payload):
    fitted, p = payload
    enc = p["encodings"]
    for key in ("sector_ppsf", "sector_ppsf_mean", "sector_ppsf_std",
                "sector_count", "society_ppsf"):
        assert enc[key], f"{key} is empty"
        assert set(fitted.encoders.__dict__[key]) == set(enc[key])

def test_encodings_carry_balcony_codes(payload):
    _, p = payload
    assert p["encodings"]["balcony"] == {"0": 0, "1": 1, "2": 2, "3": 3, "3+": 4}


# ── both the "society known" and "society unknown" metrics ship ───────────

def test_metrics_carry_both_society_known_and_unknown_numbers(payload):
    """The dashboard/docs must be able to state both honestly -- see
    homecast.model.SOCIETY_MASK_FRACTION."""
    fitted, p = payload
    m = p["metrics"]
    assert set(m["model_no_society"]) == {"mae_lakh", "mape_pct", "r2"}
    # exactly the same numbers train_final()/evaluate() computed -- the
    # payload must not recompute or otherwise drift from them
    assert m["model_no_society"] == fitted.metrics["model_no_society"]
    # (directional "withholding society can't help" is asserted properly in
    # test_model.py against fixtures with real society signal -- clean_fixture
    # here has a single constant society, so it's noise-dominated and not a
    # fair place to assert that direction)


# ── the accurate model must never be exported to the browser ──────────────

def test_accurate_model_export_raises_a_clear_error(clean_fixture):
    fitted = train_final(clean_fixture, model="accurate")
    with pytest.raises(ValueError, match="cannot be exported"):
        export_city(fitted, clean_fixture, get_city("gurgaon"))

def test_default_model_still_exports(clean_fixture):
    """The exportability guard must not false-positive on the shipped model."""
    fitted = train_final(clean_fixture, model="default")
    p = export_city(fitted, clean_fixture, get_city("gurgaon"))
    assert p["model"]["trees"]


# --- Exporting a city whose feature set is shorter -------------------------

def _sparse_city_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    area = rng.uniform(500, 2500, n).round()
    ppsf = rng.uniform(4000, 15000, n).round()
    return pd.DataFrame({
        "sector": rng.choice(["andheri", "bandra", "powai"], n),
        "area": area,
        "bedrooms": rng.integers(1, 5, n),
        "price": area * ppsf / 1e7,
        "price_per_sqft": ppsf,
        "amenity_count": rng.integers(0, 20, n).astype(float),
        "is_resale": rng.integers(0, 2, n).astype(float),
    })


@pytest.fixture()
def sparse_payload():
    df = _sparse_city_frame()
    fitted = train_final(df)
    city = City("sparsetown", "Sparse Town", Path("/x"), Path("/y"),
                Path("/z"), public=False)
    return fitted, df, export_city(fitted, df, city)


def test_feature_order_is_the_city_s_own_list_not_the_catalogue(sparse_payload):
    """The browser builds its feature row by iterating feature_order. If the
    export shipped the full catalogue for a city that trained on six columns,
    every prediction in the page would be built from the wrong vector."""
    fitted, _, p = sparse_payload
    assert p["feature_order"] == list(fitted.columns)
    assert "society_ppsf" not in p["feature_order"]
    assert "amenity_count" in p["feature_order"]


def test_importances_are_keyed_by_the_city_s_own_features(sparse_payload):
    fitted, _, p = sparse_payload
    assert set(p["feature_importances"]) == set(fitted.columns)
    assert np.isclose(sum(p["feature_importances"].values()), 1.0)


def test_sparse_city_export_predicts_identically_to_sklearn(sparse_payload):
    """The tree-walker parity guarantee has to survive a shorter vector too."""
    fitted, df, p = sparse_payload
    X = build_features(df, fitted.encoders, list(fitted.columns))
    for i in range(0, len(df), 17):
        assert predict_from_export(p, X.iloc[i].tolist()) == pytest.approx(
            float(predict_price(fitted, X.iloc[[i]])[0]), rel=1e-9)


def test_sample_omits_a_column_the_city_does_not_have(sparse_payload):
    _, _, p = sparse_payload
    assert p["sample"]
    assert "property_type" not in p["sample"][0]
    assert {"sector", "bedrooms", "area", "price"} <= set(p["sample"][0])


def test_private_city_payload_disowns_the_public_narrative(sparse_payload):
    """The page's findings and methodology prose is hand-written about
    Gurgaon's numbers. Shipping it around another city's model would be
    stating figures that are simply false for that city."""
    _, _, p = sparse_payload
    assert p["narrative"] is False


def test_public_city_payload_keeps_the_narrative(payload):
    _, p = payload
    assert p["narrative"] is True


# --- area_basis reaches the page -----------------------------------------
#
# The dashboard labels its area input. Labelling a super-built-up or an
# UNKNOWN-basis city's area "Built-up area" would state, in the one place a
# user actually reads, a basis the data never established -- exactly the
# silent equivalence the ingestion layer refuses to make. So the basis
# travels with the payload.

def _basis_payload(basis):
    df = _sparse_city_frame()
    df["area_basis"] = basis
    fitted = train_final(df)
    city = City("sparsetown", "Sparse Town", Path("/x"), Path("/y"),
                Path("/z"), public=False)
    return export_city(fitted, df, city)


@pytest.mark.parametrize("basis", ["superbuiltup", "builtup", "carpet", "unknown"])
def test_payload_carries_the_city_s_declared_area_basis(basis):
    assert _basis_payload(basis)["area_basis"] == basis


def test_unknown_basis_is_carried_verbatim_not_dropped_or_defaulted():
    """"unknown" must survive as itself. Dropping the key would make the page
    fall back to the built-up wording -- silently converting "we do not know"
    into a specific claim."""
    p = _basis_payload("unknown")
    assert p["area_basis"] == "unknown"
    assert p["area_basis"] != "builtup"


def test_payload_omits_area_basis_when_the_frame_never_recorded_one(payload):
    """Gurgaon predates the ingestion config and carries no area_basis
    column. The key must then be absent rather than guessed, so the public
    payload is unchanged by this feature."""
    _, p = payload
    assert "area_basis" not in p


def test_mixed_area_basis_in_one_frame_is_refused():
    """A frame pooling two bases has no single basis to label, and its
    price_per_sqft mixes quantities that are not the same measurement."""
    df = _sparse_city_frame()
    df["area_basis"] = ["carpet" if i % 2 else "builtup" for i in range(len(df))]
    fitted = train_final(df)
    city = City("mixed", "Mixed", Path("/x"), Path("/y"), Path("/z"), public=False)
    with pytest.raises(ValueError, match="single area_basis"):
        export_city(fitted, df, city)


# --- baseline_served reaches the payload -----------------------------------
#
# A city whose model loses to its own locality-median rule must not have its
# dashboard present the model's number as THE estimate. export_city has to
# carry that verdict (and, when it applies, the plain-language reason) into
# the payload the template reads -- see BASELINE_SERVED in
# dashboard_template.html. clean_fixture (see conftest.py) is small and
# noisy enough that its model genuinely loses to the sector baseline, so
# `payload` below is a real losing case, not a mocked one; `sparse_payload`
# (120 rows, real per-locality structure) is a real winning one.

def test_losing_city_is_flagged_baseline_served(payload):
    fitted, p = payload
    assert fitted.metrics["model"]["mape_pct"] >= fitted.metrics["baseline_sector"]["mape_pct"], (
        "fixture assumption changed: this test needs clean_fixture's model "
        "to lose, to prove the payload actually carries a real losing verdict")
    assert p["baseline_served"] is True
    assert "baseline_note" in p and p["baseline_note"]


def test_winning_city_is_not_flagged_and_carries_no_note(sparse_payload):
    fitted, _, p = sparse_payload
    assert fitted.metrics["model"]["mape_pct"] < fitted.metrics["baseline_sector"]["mape_pct"], (
        "fixture assumption changed: this test needs the sparse city's model "
        "to win, to prove the flag stays off on a real winning case")
    assert p["baseline_served"] is False
    assert "baseline_note" not in p


def test_baseline_band_is_always_exported(payload, sparse_payload):
    """Exported unconditionally (like `band`), not only for a losing city --
    see the comment in export.py: a caller may want to show the rule's own
    range even when the model is the one being served."""
    _, p_lose = payload
    _, _, p_win = sparse_payload
    for p in (p_lose, p_win):
        assert "baseline_band" in p and len(p["baseline_band"]) == 2
        assert p["baseline_band"][0] <= p["baseline_band"][1]


# --- quality floor / serving_status reach the payload (Correction 2) ------
#
# A city where even the BETTER of {model, baseline} is still too unreliable
# must not have its dashboard show a headline price at all. serving_status
# is the general three-way decision; the export payload has to carry it (and
# the floor value itself, and a plain-language reason when not_served) so
# the template can gate on it. clean_fixture (via `payload`) is a real
# not_served case: noisy enough that both numbers clear the 30% floor.

def test_quality_floor_mape_pct_is_always_exported(payload):
    from homecast.model import QUALITY_FLOOR_MAPE_PCT
    _, p = payload
    assert p["quality_floor_mape_pct"] == QUALITY_FLOOR_MAPE_PCT


def test_not_served_city_carries_serving_status_and_note(payload):
    fitted, p = payload
    assert fitted.metrics["serving_status"] == "not_served", (
        "fixture assumption changed: this test needs clean_fixture to be "
        "genuinely hopeless under the quality floor, to prove the payload "
        "carries a real not_served verdict")
    assert p["serving_status"] == "not_served"
    assert "not_served_note" in p and p["not_served_note"]
    assert "not good enough to price a property responsibly" in p["not_served_note"]


def _real_model_served_frame() -> pd.DataFrame:
    """A real, non-rigged model win clearly UNDER the quality floor (unlike
    sparse_payload's fixture below, which really does beat its own baseline
    but both numbers land above 30% MAPE -- see the not_served test below)."""
    rng = np.random.default_rng(5)
    rows = []
    base_rate = {f"loc{i}": rng.uniform(4000, 15000) for i in range(8)}
    for loc, rate in base_rate.items():
        for _ in range(45):
            area = rng.uniform(500, 2500)
            bedrooms = int(rng.integers(1, 6))
            ppsf = rate * (1 + 0.18 * (bedrooms - 3)) * rng.uniform(0.98, 1.02)
            rows.append((loc, area, bedrooms, ppsf * area / 1e7, ppsf))
    return pd.DataFrame(rows, columns=["sector", "area", "bedrooms", "price", "price_per_sqft"])


def test_real_model_served_payload_carries_serving_status_model():
    df = _real_model_served_frame()
    fitted = train_final(df)
    city = City("modelservedtown", "Model Served Town", Path("/x"), Path("/y"),
               Path("/z"), public=False)
    p = export_city(fitted, df, city)
    assert p["serving_status"] == "model"
    assert "not_served_note" not in p


def test_a_real_win_that_still_fails_the_floor_is_not_served(sparse_payload):
    """sparse_payload's model genuinely beats its own baseline (see
    test_winning_city_is_not_flagged_and_carries_no_note above) -- but both
    numbers sit above the 30% floor. baseline_served stays False (unchanged,
    two-way meaning) while serving_status must still be not_served: proof the
    floor is a real, independent gate, not just a rename of baseline_served."""
    fitted, _, p = sparse_payload
    assert fitted.metrics["model"]["mape_pct"] >= 30.0
    assert fitted.metrics["baseline_sector"]["mape_pct"] >= 30.0
    assert p["baseline_served"] is False
    assert p["serving_status"] == "not_served"
    assert "not_served_note" in p


def _real_baseline_served_frame() -> pd.DataFrame:
    """A real, non-rigged baseline win clearly UNDER the quality floor:
    bedrooms carry no real price signal, so the model has nothing to add
    over an already-accurate locality median."""
    rng = np.random.default_rng(3)
    rows = []
    base_rate = {f"loc{i}": rng.uniform(5000, 12000) for i in range(6)}
    for loc, rate in base_rate.items():
        for _ in range(60):
            area = rng.uniform(600, 2200)
            bedrooms = int(rng.integers(1, 6))
            ppsf = rate * float(np.exp(rng.normal(0.0, 0.06)))
            rows.append((loc, area, bedrooms, ppsf * area / 1e7, ppsf))
    return pd.DataFrame(rows, columns=["sector", "area", "bedrooms", "price", "price_per_sqft"])


def test_real_baseline_served_payload_carries_serving_status_baseline():
    df = _real_baseline_served_frame()
    fitted = train_final(df)
    city = City("baselineservedtown", "Baseline Served Town", Path("/x"),
               Path("/y"), Path("/z"), public=False)
    p = export_city(fitted, df, city)
    assert p["serving_status"] == "baseline"
    assert p["baseline_served"] is True
    assert "baseline_note" in p and p["baseline_note"]
    assert "not_served_note" not in p
