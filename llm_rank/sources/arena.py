"""LMArena Chatbot Arena leaderboard via HuggingFace dataset mirror.

Covers proprietary frontier models (Claude, GPT, Gemini, Grok) that HF Open LLM
Leaderboard never scores. Provides Elo (~800–1500) which we normalize to 0–100
so it can stack against HF v2 averages.

Dataset: mathewhe/chatbot-arena-elo (218 rows, updated daily).
"""

from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

DATASET = "mathewhe/chatbot-arena-elo"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
MAX_ROWS = 1000

# Observed Arena Score range: ~800 (weak) to ~1520 (frontier).
ELO_FLOOR = 800.0
ELO_CEIL = 1520.0


def _normalize_elo(elo: float | None) -> float | None:
    if elo is None:
        return None
    pct = (elo - ELO_FLOOR) / (ELO_CEIL - ELO_FLOOR)
    return max(0.0, min(1.0, pct)) * 100.0


def _fetch_page(offset: int, length: int) -> list[dict[str, Any]]:
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }
    r = requests.get(ROWS_URL, params=params, timeout=30)
    r.raise_for_status()
    return [item.get("row", {}) for item in r.json().get("rows", [])]


def fetch_leaderboard() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    while offset < MAX_ROWS:
        try:
            rows = _fetch_page(offset, PAGE_SIZE)
        except requests.RequestException as e:
            log.warning("Arena fetch failed at offset %d: %s", offset, e)
            break
        if not rows:
            break
        for r in rows:
            name = r.get("Model")
            elo = r.get("Arena Score")
            if not name or elo is None:
                continue
            quality = _normalize_elo(float(elo))
            out.append({
                "model_id": str(name),
                "display_name": str(name),
                "quality": {
                    "general": quality,
                    "coding": quality,
                    "math": quality,
                    "reasoning": quality,
                },
                "elo": float(elo),
                "organization": r.get("Organization"),
                "license": r.get("License"),
                "votes": r.get("Votes"),
                "source": "arena",
            })
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return out
