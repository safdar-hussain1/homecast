import numpy as np
import pytest

from homecast.model import train_final
from homecast.valuation import Query, comparables, estimate


@pytest.fixture()
def fitted(clean_fixture):
    return train_final(clean_fixture)

def q(**over):
    base = dict(sector="sector 1", property_type="flat", bedrooms=3,
                bathrooms=2, area=1500.0, furnishing="semi-furnished",
                luxury_score=50)
    base.update(over)
    return Query(**base)

def test_estimate_band_applies_the_stored_band_with_the_right_sign(fitted):
    """band = (q10, q90) of log(pred) - log(actual), so actual = pred*exp(-res):
    the q90 residual sets the LOW end and q10 the HIGH end. A sign flip here
    would still produce lo < price < hi, so assert the exact multipliers."""
    e = estimate(fitted, q())
    q10, q90 = fitted.band
    assert e["lo_cr"] == pytest.approx(e["price_cr"] * np.exp(-q90))
    assert e["hi_cr"] == pytest.approx(e["price_cr"] * np.exp(-q10))
    assert e["lo_cr"] < e["price_cr"] < e["hi_cr"]
    assert e["price_cr"] > 0

def test_bigger_area_never_cheaper_much(fitted):
    small, big = estimate(fitted, q(area=600.0)), estimate(fitted, q(area=2900.0))
    assert big["price_cr"] > small["price_cr"]

def test_out_of_range_area_rejected(fitted):
    with pytest.raises(ValueError, match="area"):
        estimate(fitted, q(area=10.0))

def test_bad_property_type_rejected(fitted):
    with pytest.raises(ValueError, match="property_type"):
        estimate(fitted, q(property_type="castle"))

def test_comparables_same_sector_first(clean_fixture):
    c = comparables(clean_fixture, q(), k=3)
    assert len(c) == 3
    assert (c["sector"] == "sector 1").all()

def test_comparables_falls_back_when_sector_unknown(clean_fixture):
    c = comparables(clean_fixture, q(sector="sector 99"), k=3)
    assert len(c) == 3
    # nearest-by-area across the whole frame, since no listing is in sector 99
    expected = (clean_fixture["area"] - 1500.0).abs().nsmallest(3).index
    assert set(c.index) == set(expected)

def test_bad_furnishing_rejected(fitted):
    with pytest.raises(ValueError, match="furnishing"):
        estimate(fitted, q(furnishing="gold-plated"))
