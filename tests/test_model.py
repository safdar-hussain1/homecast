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

def test_band_is_ordered(clean_fixture):
    f = train_final(clean_fixture)
    lo, hi = f.band
    assert lo < 0 < hi or lo < hi   # 10th < 90th percentile always

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
