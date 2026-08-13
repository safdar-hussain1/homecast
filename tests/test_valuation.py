import numpy as np
import pandas as pd
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

def test_unknown_sector_rejected(fitted):
    """A case typo must fail loudly, not fall back to the global median."""
    with pytest.raises(ValueError, match="Unknown sector 'Sector 1'"):
        estimate(fitted, q(sector="Sector 1"))

def test_global_fallback_key_is_not_a_usable_sector(fitted):
    with pytest.raises(ValueError, match="Unknown sector"):
        estimate(fitted, q(sector="__global__"))

def test_unknown_age_rejected(fitted):
    """An unrecognised age string used to be silently coded as 'missing'."""
    with pytest.raises(ValueError, match="Unknown age 'Brand New'"):
        estimate(fitted, q(age="Brand New"))

def test_age_none_is_allowed(fitted):
    assert estimate(fitted, q(age=None))["price_cr"] > 0

@pytest.mark.parametrize("field,value", [
    ("bedrooms", 99),
    ("bathrooms", 99),
    ("luxury_score", 5000),
])
def test_out_of_range_numeric_inputs_rejected(fitted, field, value):
    with pytest.raises(ValueError, match=f"{field} {value} outside supported range"):
        estimate(fitted, q(**{field: value}))


# ── optional society / balcony inputs ──────────────────────────────────────

def test_society_and_balcony_are_optional(fitted):
    """Omitting both must still produce a normal estimate (society falls
    back to the listing's own sector rate, balcony to the missing code),
    not an error."""
    e = estimate(fitted, q())
    assert e["price_cr"] > 0

def test_unknown_balcony_rejected(fitted):
    with pytest.raises(ValueError, match="Unknown balcony '9'"):
        estimate(fitted, q(balcony="9"))

def test_valid_balcony_accepted(fitted):
    assert estimate(fitted, q(balcony="2"))["price_cr"] > 0

def test_unrecognised_society_falls_back_instead_of_erroring(fitted):
    """Unlike sector, an unknown society is not a typo trap -- it legitimately
    means "not one of the societies seen in training" and must fall back
    quietly, the same way an unseen sector does inside a CV fold."""
    e = estimate(fitted, q(society="a society nobody has ever heard of"))
    assert e["price_cr"] > 0

@pytest.fixture()
def society_signal_fixture() -> pd.DataFrame:
    """Two societies with clearly separated ₹/sq.ft. (unlike the shared
    `clean_fixture`, which has a single constant society and so carries no
    society signal at all) -- enough for a fitted model to have actually
    learned something from `society_ppsf`."""
    rng = np.random.default_rng(9)
    n = 200
    # Deliberately imbalanced (70/30) so the dataset's global median sits
    # inside the majority ("low society") cluster -- that makes the
    # assertions below robust to exactly where the median falls, rather than
    # relying on a ~50/50 random split landing on one side by chance.
    society = np.array(["low society"] * 140 + ["high society"] * 60)
    base = np.where(society == "high society", 20000.0, 6000.0)
    ppsf = base * np.exp(rng.normal(0.0, 0.03, n))
    area = rng.uniform(800.0, 2000.0, n)
    return pd.DataFrame({
        "sector": "sector 1", "society": society, "property_type": "flat",
        "price": area * ppsf / 1e7, "price_per_sqft": ppsf, "area": area,
        "bedrooms": 3, "bathrooms": 2, "balcony": "2",
        "furnishing_type": "semi-furnished", "luxury_score": 50,
        "age_possession": "New Property",
    })

# --- served_by: which number is THE estimate ------------------------------
#
# A city whose model does not beat its own locality-median rule must have
# the rule presented as the estimate, not the model's own number (see
# homecast.model.model_beats_baseline). clean_fixture's model (see
# conftest.py) is small and noisy enough that it genuinely loses to the
# sector baseline -- this is a real losing case, not a mocked metrics dict.

def test_losing_model_is_served_by_the_baseline(fitted):
    from homecast.model import model_beats_baseline
    assert not model_beats_baseline(fitted.metrics), (
        "fixture assumption changed: this test needs clean_fixture's model "
        "to lose, to prove estimate() actually falls back for a real case")
    e = estimate(fitted, q())
    assert e["served_by"] == "baseline"
    assert e["price_cr"] > 0                       # the model number is still there...
    assert "note" in e and e["note"]
    # ...but baseline_price_cr, not price_cr, is what a caller must present
    # as THE estimate -- see cli.py's predict command, which reads this.
    assert e["baseline_price_cr"] > 0
    assert e["baseline_lo_cr"] < e["baseline_price_cr"] < e["baseline_hi_cr"]


@pytest.fixture()
def winning_fixture():
    """A real, non-rigged case where the model beats the baseline: several
    localities, each with its own base rate, plus a genuine bedroom-driven
    ₹/sq.ft. effect the locality-median rule cannot see (it only ever
    multiplies one number by area)."""
    rng = np.random.default_rng(5)
    rows = []
    base_rate = {f"loc{i}": rng.uniform(4000, 15000) for i in range(8)}
    for loc, rate in base_rate.items():
        for _ in range(45):
            area = rng.uniform(500, 2500)
            bedrooms = int(rng.integers(1, 6))
            ppsf = rate * (1 + 0.18 * (bedrooms - 3)) * rng.uniform(0.98, 1.02)
            rows.append((loc, area, bedrooms, ppsf * area / 1e7, ppsf))
    df = pd.DataFrame(rows, columns=["sector", "area", "bedrooms", "price", "price_per_sqft"])
    return train_final(df)


def test_winning_model_is_served_by_itself(winning_fixture):
    from homecast.model import model_beats_baseline
    assert model_beats_baseline(winning_fixture.metrics), (
        "fixture assumption changed: this test needs the model to win, to "
        "prove served_by stays 'model' on a real winning case")
    e = estimate(winning_fixture, Query(sector="loc0", property_type="flat",
                                        bedrooms=3, bathrooms=2, area=1200.0,
                                        furnishing="semi-furnished", luxury_score=50))
    assert e["served_by"] == "model"
    assert "note" not in e
    # the model's own band is unaffected by served_by -- still price_cr's band
    assert e["lo_cr"] < e["price_cr"] < e["hi_cr"]


def test_known_society_changes_the_prediction(society_signal_fixture):
    """society_ppsf is a real, informative feature: naming the actual society
    must move the prediction toward that society's price level (otherwise the
    feature would be dead weight at inference time)."""
    f = train_final(society_signal_fixture)
    base = dict(sector="sector 1", property_type="flat", bedrooms=3, bathrooms=2,
                area=1400.0, furnishing="semi-furnished", luxury_score=50)
    high = estimate(f, Query(**base, society="high society"))["price_cr"]
    low = estimate(f, Query(**base, society="low society"))["price_cr"]
    unspecified = estimate(f, Query(**base))["price_cr"]
    assert high > low
    # naming the pricier society must lift the estimate above the
    # unspecified (sector-rate-fallback) default
    assert high > unspecified
