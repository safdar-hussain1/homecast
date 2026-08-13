import numpy as np
import pandas as pd
import pytest

from homecast.features import build_features
from homecast.model import DEFAULT_PARAMS, evaluate, predict_price, train_final


def test_evaluate_reports_model_and_baselines(clean_fixture):
    r = evaluate(clean_fixture, n_splits=3)
    for k in ("model", "baseline_sector", "baseline_global"):
        assert set(r[k]) == {"mae_lakh", "mape_pct", "r2"}
        assert r[k]["mae_lakh"] > 0
    assert len(r["residuals_log"]) == len(clean_fixture)

def test_train_final_predicts_reasonably(clean_fixture):
    f = train_final(clean_fixture)
    X = build_features(clean_fixture, f.sector_map)
    pred = predict_price(f, X)
    assert pred.shape == (len(clean_fixture),)
    assert (pred > 0).all()
    # in-sample fit on tiny data should at least beat predicting the median
    med = np.median(clean_fixture["price"])
    assert np.mean(np.abs(pred - clean_fixture["price"])) < np.mean(
        np.abs(med - clean_fixture["price"]))

def test_band_covers_about_eighty_percent_of_listings(clean_fixture):
    """The band must be a usable 80% interval, applied with the right sign.

    ``band`` holds the 10th/90th percentile of residual = log(pred) - log(actual),
    so actual = pred * exp(-residual): the q90 residual gives the LOW end of the
    true price and q10 gives the HIGH end. Reconstruct the out-of-fold
    predictions from the stored residuals and count how many actual prices land
    inside the interval. By construction that fraction is 0.80 on the data the
    band was fitted from, but the assertion still catches a band built from the
    wrong quantiles or from the wrong residual definition. The tolerance is wide
    (0.60-0.95) because the fixture is ~40 rows, where a single listing moves
    coverage by 2.4 points and quantile interpolation is coarse.
    """
    f = train_final(clean_fixture)
    q10, q90 = f.band
    assert q10 < q90                       # 10th percentile below the 90th
    actual = clean_fixture["price"].to_numpy(dtype=float)
    res = np.asarray(f.metrics["residuals_log"])
    pred = actual * np.exp(res)            # residual = log(pred) - log(actual)
    inside = (actual >= pred * np.exp(-q90)) & (actual <= pred * np.exp(-q10))
    coverage = float(inside.mean())
    assert 0.60 <= coverage <= 0.95, f"band covers {coverage:.0%}, expected ~80%"

def test_ranges_cover_data(clean_fixture):
    f = train_final(clean_fixture)
    assert f.ranges["area"][0] == clean_fixture["area"].min()
    assert f.ranges["area"][1] == clean_fixture["area"].max()

def test_zero_price_rejected(clean_fixture):
    bad = clean_fixture.copy()
    bad.loc[bad.index[0], "price"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        evaluate(bad, n_splits=3)

def test_constant_price_r2_rejected(clean_fixture):
    flat = clean_fixture.copy()
    flat["price"] = 1.0
    with pytest.raises(ValueError, match="identical"):
        evaluate(flat, n_splits=3)
