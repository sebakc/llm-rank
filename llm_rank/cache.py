"""JSON file cache with TTL."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

CACHE_DIR = Path(os.environ.get("LLM_RANK_CACHE_DIR", str(Path.home() / ".cache" / "llm-rank")))


def _path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def is_fresh(key: str, ttl_seconds: int) -> bool:
    p = _path(key)
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < ttl_seconds


def read(key: str) -> Any | None:
    p = _path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write(key: str, value: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(key).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_path(key))


def get(
    key: str,
    loader: Callable[[], Any],
    ttl_hours: float = 24.0,
    refresh: bool = False,
) -> Any:
    """Return cached value or load via `loader`, persist, and return."""
    ttl_s = int(ttl_hours * 3600)
    if not refresh and is_fresh(key, ttl_s):
        cached = read(key)
        if cached is not None:
            return cached
    value = loader()
    write(key, value)
    return value


def clear(key: str | None = None) -> None:
    if key is None:
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()
        return
    p = _path(key)
    if p.exists():
        p.unlink()
