import numpy as np
import pandas as pd
import pytest

from homecast.features import (FEATURE_COLUMNS, build_features,
                               sector_encoding, target)


def test_feature_matrix_shape_and_columns(clean_fixture):
    m = sector_encoding(clean_fixture)
    X = build_features(clean_fixture, m)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(clean_fixture)
    assert X.notna().all().all()

def test_sector_encoding_is_train_only(clean_fixture):
    train = clean_fixture[clean_fixture["sector"] != "sector 3"]
    m = sector_encoding(train)
    assert "sector 3" not in m                     # never saw it
    X = build_features(clean_fixture, m)           # applied to full data
    unseen = clean_fixture["sector"] == "sector 3"
    assert (X.loc[unseen, "sector_ppsf"] == m["__global__"]).all()

def test_sector_encoding_values(clean_fixture):
    m = sector_encoding(clean_fixture)
    s1 = clean_fixture[clean_fixture["sector"] == "sector 1"]
    assert m["sector 1"] == pytest.approx(s1["price_per_sqft"].median())

def test_target_is_log_price(clean_fixture):
    assert np.allclose(target(clean_fixture), np.log(clean_fixture["price"]))

def test_unknown_furnishing_fails_fast(clean_fixture):
    bad = clean_fixture.copy()
    bad.loc[bad.index[0], "furnishing_type"] = "gold-plated"
    with pytest.raises(ValueError, match="furnishing"):
        build_features(bad, sector_encoding(clean_fixture))

def test_missing_age_coded_minus_one(clean_fixture):
    m = sector_encoding(clean_fixture)
    X = build_features(clean_fixture, m)
    missing = clean_fixture["age_possession"].isna()
    if missing.any():
        assert (X.loc[missing, "age_code"] == -1).all()
