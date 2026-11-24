from pathlib import Path

from whodat.paths import cache_dir, state_dir


def test_cache_dir_follows_xdg_cache_home():
    assert cache_dir({"XDG_CACHE_HOME": "/x/cache"}) == Path("/x/cache/whodat")


def test_state_dir_defaults_under_home_local_state():
    assert state_dir({"HOME": "/home/u"}) == Path("/home/u/.local/state/whodat")
