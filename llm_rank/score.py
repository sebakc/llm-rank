"""Scoring + ranking logic."""

from __future__ import annotations

import math
from typing import Iterable

from .models import Constraints, ModelEntry, Task


def quality_for(entry: ModelEntry, task: Task) -> float | None:
    """Best available quality score for a task, fall back to general."""
    q = entry.quality.get(task)
    if q is None:
        q = entry.quality.get("general")
    return q


def normalize_quality(q: float | None) -> float:
    if q is None:
        return 0.0
    # HF leaderboard scores are 0..100. Clamp + scale.
    return max(0.0, min(1.0, q / 100.0))


def normalize_price(price: float | None, max_seen: float | None) -> float:
    """Lower is better. 0 (cheapest) -> 0.0; max_seen -> 1.0. Log-scaled."""
    if price is None or max_seen is None or max_seen <= 0:
        return 0.5  # neutral when missing
    if price <= 0:
        return 0.0
    return math.log1p(price) / math.log1p(max_seen)


def normalize_speed(tps: float | None, max_seen: float | None) -> float:
    if tps is None or max_seen is None or max_seen <= 0:
        return 0.0
    return max(0.0, min(1.0, tps / max_seen))


def score_entry(
    entry: ModelEntry,
    c: Constraints,
    max_price: float | None,
    max_tps: float | None,
) -> float:
    w = c.weights
    q = normalize_quality(quality_for(entry, c.task))
    p = normalize_price(entry.price_avg, max_price)
    s = normalize_speed(entry.throughput_tps, max_tps)
    return q * w.get("quality", 1.0) - p * w.get("cost", 0.5) + s * w.get("speed", 0.3)


def rank(entries: Iterable[ModelEntry], c: Constraints) -> list[tuple[ModelEntry, float]]:
    """Rank entries. Rated models (with HF quality) come first, sorted by score.
    Unrated models come after, sorted by price ascending so cheap frontier
    models (e.g. DeepSeek-V4 with no HF eval yet) are still visible."""
    entries = list(entries)
    prices = [e.price_avg for e in entries if e.price_avg is not None and e.price_avg > 0]
    tpses = [e.throughput_tps for e in entries if e.throughput_tps is not None and e.throughput_tps > 0]
    max_price = max(prices) if prices else None
    max_tps = max(tpses) if tpses else None

    rated: list[tuple[ModelEntry, float]] = []
    unrated: list[tuple[ModelEntry, float]] = []
    for e in entries:
        s = score_entry(e, c, max_price, max_tps)
        if quality_for(e, c.task) is None:
            unrated.append((e, s))
        else:
            rated.append((e, s))
    rated.sort(key=lambda x: x[1], reverse=True)
    # Unrated: cheapest first; None price sinks.
    unrated.sort(key=lambda x: (x[0].price_avg is None, x[0].price_avg or 0.0))
    return rated + unrated
