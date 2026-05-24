"""OpenRouter frontend (undocumented) API fetcher.

Provides the data the official `/api/v1/*` endpoints leave blank:
- full model catalog (~754 models incl. hidden, vs ~358 public)
- per-provider p50–p99 throughput (tok/s) + latency (ms TTFT)
- uptime % and success/error counts

Routes discovered by grepping OR's compiled JS chunks; documented here so we
notice if they break.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

log = logging.getLogger(__name__)

MODELS_URL = "https://openrouter.ai/api/frontend/models"
STATS_URL = "https://openrouter.ai/api/frontend/stats/endpoint"
TIMEOUT = 30
UA = "llm-rank/0.1 (+https://github.com/yourname/llm-rank)"


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_catalog() -> list[dict[str, Any]]:
    """Full model catalog from the frontend API."""
    try:
        r = requests.get(MODELS_URL, timeout=TIMEOUT, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.RequestException as e:
        log.warning("openrouter frontend catalog fetch failed: %s", e)
        return []


def fetch_stats(permaslug: str) -> list[dict[str, Any]]:
    """Per-provider perf stats for a single model permaslug."""
    try:
        r = requests.get(
            STATS_URL,
            params={"permaslug": permaslug},
            timeout=TIMEOUT,
            headers={"User-Agent": UA},
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.RequestException as e:
        log.debug("stats fetch failed for %s: %s", permaslug, e)
        return []


def _aggregate_providers(provider_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-provider rows into a summary + keep the raw per-provider list."""
    speeds = []
    latencies = []
    uptimes = []
    providers = []
    provider_pricing: list[dict[str, Any]] = []
    best_price_in = None
    best_price_out = None
    best_provider = None
    best_throughput = None

    for ep in provider_rows:
        stats = ep.get("stats") or {}
        tp = _to_float(stats.get("p50_throughput"))
        lat = _to_float(stats.get("p50_latency"))
        reqs = stats.get("request_count")
        pricing = ep.get("pricing") or {}
        p_in = _to_float(pricing.get("prompt"))
        p_out = _to_float(pricing.get("completion"))
        p_cache = _to_float(pricing.get("input_cache_read"))
        up = _to_float(ep.get("uptime_last_30m"))
        provider_name = ep.get("provider_name") or ep.get("provider_display_name")

        if tp is not None:
            speeds.append(tp)
            if best_throughput is None or tp > best_throughput:
                best_throughput = tp
                best_provider = provider_name
        if lat is not None:
            latencies.append(lat)
        if up is not None:
            uptimes.append(up)
        if p_in is not None:
            best_price_in = p_in if best_price_in is None else min(best_price_in, p_in)
        if p_out is not None:
            best_price_out = p_out if best_price_out is None else min(best_price_out, p_out)
        if provider_name:
            providers.append(provider_name)
            provider_pricing.append({
                "provider": provider_name,
                "price_in_per_m": p_in * 1_000_000 if p_in is not None else None,
                "price_out_per_m": p_out * 1_000_000 if p_out is not None else None,
                "price_cache_per_m": p_cache * 1_000_000 if p_cache is not None else None,
                "throughput_tps": tp,
                "latency_ms": lat,
                "uptime_pct": up,
                "request_count": reqs,
                "quantization": ep.get("quantization"),
                "context_length": ep.get("context_length"),
            })

    # Sort providers by cheapest avg price ascending (then by throughput desc).
    def _key(pp: dict[str, Any]) -> tuple[float, float]:
        pi = pp.get("price_in_per_m") or 0.0
        po = pp.get("price_out_per_m") or 0.0
        avg = (pi + po) / 2 if (pi or po) else 999.0
        tp = -(pp.get("throughput_tps") or 0.0)
        return (avg, tp)
    provider_pricing.sort(key=_key)

    return {
        "throughput_p50": max(speeds) if speeds else None,
        "throughput_median_p50": sorted(speeds)[len(speeds) // 2] if speeds else None,
        "latency_p50_ms": min(latencies) if latencies else None,
        "uptime_pct": max(uptimes) if uptimes else None,
        "providers": providers,
        "provider_pricing": provider_pricing,
        "best_provider": best_provider,
        # OR pricing is per-token; convert to $/M tokens.
        "price_in_per_m": best_price_in * 1_000_000 if best_price_in is not None else None,
        "price_out_per_m": best_price_out * 1_000_000 if best_price_out is not None else None,
    }


def _build_row(m: dict[str, Any]) -> dict[str, Any] | None:
    permaslug = m.get("permaslug") or m.get("endpoint", {}).get("model_variant_permaslug")
    slug = m.get("slug")
    if not slug:
        return None
    # Drop non-text models (image/video/audio generators).
    out_mods = m.get("output_modalities") or []
    if out_mods and "text" not in out_mods:
        return None
    # Drop routers / hidden / disabled / deranked entries.
    if m.get("hidden") or m.get("router"):
        return None
    providers = fetch_stats(permaslug) if permaslug else []
    agg = _aggregate_providers(providers)

    # Skip routers and pseudo-models with negative sentinel pricing
    if (agg["price_in_per_m"] is not None and agg["price_in_per_m"] < 0) or (
        agg["price_out_per_m"] is not None and agg["price_out_per_m"] < 0
    ):
        return None

    return {
        "model_id": slug,
        "display_name": m.get("name") or slug,
        "hf_slug": m.get("hf_slug"),
        "context_len": m.get("context_length"),
        "created_at": m.get("created_at"),
        "input_modalities": m.get("input_modalities"),
        "output_modalities": m.get("output_modalities"),
        **agg,
    }


def fetch_all(progress: bool = False, concurrency: int = 16) -> list[dict[str, Any]]:
    """Fetch catalog + per-model perf. Returns one normalized row per model.

    Heavy: ~750 HTTP calls fanned out across `concurrency` workers. Cached by callers.
    """
    catalog = fetch_catalog()
    n = len(catalog)
    out: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_build_row, m): m for m in catalog}
        for fut in as_completed(futures):
            row = fut.result()
            done += 1
            if progress and done % 50 == 0:
                log.warning("openrouter frontend: %d/%d models hydrated", done, n)
            if row is not None:
                out.append(row)
    return out
