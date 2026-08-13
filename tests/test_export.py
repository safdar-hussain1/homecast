import json

import numpy as np
import pandas as pd
import pytest

from homecast.cities import get_city
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
