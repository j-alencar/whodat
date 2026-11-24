from whodat.labels import Region
from whodat.pipeline import windows


def test_long_region_splits_into_equal_windows_within_limit():
    assert windows([Region(0.0, 2400.0), Region(3000.0, 3600.0)], 1000.0) == [
        Region(0.0, 800.0),
        Region(800.0, 1600.0),
        Region(1600.0, 2400.0),
        Region(3000.0, 3600.0),
    ]
