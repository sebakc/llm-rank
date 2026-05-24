import time

from llm_rank import cache


def test_cache_get_writes_then_reads(isolated_cache):
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return {"hello": "world"}

    v1 = cache.get("k1", loader, ttl_hours=1.0)
    v2 = cache.get("k1", loader, ttl_hours=1.0)
    assert v1 == v2 == {"hello": "world"}
    assert calls["n"] == 1  # second call hit cache


def test_cache_refresh_bypasses_ttl(isolated_cache):
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return [calls["n"]]

    cache.get("k2", loader, ttl_hours=24.0)
    cache.get("k2", loader, ttl_hours=24.0, refresh=True)
    assert calls["n"] == 2


def test_cache_expires(isolated_cache):
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return calls["n"]

    cache.get("k3", loader, ttl_hours=1.0)
    # Backdate file mtime to force expiry.
    p = isolated_cache / "k3.json"
    old = time.time() - 60 * 60 * 25
    import os
    os.utime(p, (old, old))
    cache.get("k3", loader, ttl_hours=24.0)
    assert calls["n"] == 2


def test_clear(isolated_cache):
    cache.write("k4", {"a": 1})
    assert cache.read("k4") == {"a": 1}
    cache.clear("k4")
    assert cache.read("k4") is None
