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

def test_estimate_band_ordering(fitted):
    e = estimate(fitted, q())
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
