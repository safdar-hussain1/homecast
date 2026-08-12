import json

import numpy as np
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
    X = build_features(clean_fixture, fitted.sector_map)
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

def test_no_nan_in_payload(payload, tmp_path):
    _, p = payload
    write_export(p, tmp_path / "x.json")   # allow_nan=False raises on NaN

def test_residual_hist_shape(payload):
    _, p = payload
    assert len(p["residual_hist"]["edges"]) == 21
    assert len(p["residual_hist"]["counts"]) == 20
