"""OpenRouter models endpoint fetcher."""

from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

URL = "https://openrouter.ai/api/v1/models"


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize(item: dict[str, Any]) -> dict[str, Any] | None:
    model_id = item.get("id")
    if not model_id:
        return None
    pricing = item.get("pricing") or {}
    # OpenRouter prices are per-token strings. Convert to $/M tokens.
    p_in = _to_float(pricing.get("prompt"))
    p_out = _to_float(pricing.get("completion"))
    # Filter out router placeholders that report negative pricing.
    if (p_in is not None and p_in < 0) or (p_out is not None and p_out < 0):
        return None
    price_in_m = p_in * 1_000_000 if p_in is not None else None
    price_out_m = p_out * 1_000_000 if p_out is not None else None
    name = item.get("name") or model_id
    return {
        "model_id": model_id,
        "display_name": name,
        "price_in": price_in_m,
        "price_out": price_out_m,
        "context_len": item.get("context_length"),
        "provider": "openrouter",
        "available_cloud": True,
    }


def fetch_models() -> list[dict[str, Any]]:
    try:
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
    except requests.RequestException as e:
        log.warning("OpenRouter fetch failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        norm = _normalize(item)
        if norm is not None:
            out.append(norm)
    return out
