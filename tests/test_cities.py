import pytest
from homecast.cities import get_city

def test_gurgaon_registered():
    c = get_city("gurgaon")
    assert c.display == "Gurgaon"
    assert c.raw_path.name == "gurgaon_properties.csv"
    assert c.raw_path.exists()

def test_unknown_city_lists_valid_keys():
    with pytest.raises(ValueError, match="gurgaon"):
        get_city("atlantis")
