"""HuggingFace Open LLM Leaderboard fetcher (datasets-server REST)."""

from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

DATASET = "open-llm-leaderboard/contents"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
MAX_ROWS = 8000  # safety cap


# Column name -> normalized task key. Order matters: first match wins per task.
_TASK_COLUMNS: dict[str, list[str]] = {
    "general": ["Average ⬆️", "Average", "average"],
    "math": ["MATH Lvl 5", "math", "gsm8k"],
    "reasoning": ["BBH", "bbh", "arc_challenge", "MUSR", "GPQA"],
    "coding": ["HumanEval", "humaneval", "mbpp"],  # not in v2; falls back to general
}


def _pick(row: dict[str, Any], keys: list[str]) -> float | None:
    for k in keys:
        if k in row and row[k] is not None:
            try:
                v = float(row[k])
                if v != v:  # NaN
                    continue
                return v
            except (TypeError, ValueError):
                continue
    return None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    name = (
        row.get("fullname")
        or row.get("Model")
        or row.get("eval_name")
        or row.get("model")
    )
    if not name:
        return None
    quality = {}
    for task, cols in _TASK_COLUMNS.items():
        v = _pick(row, cols)
        if v is not None:
            quality[task] = v
    if "general" not in quality:
        return None  # skip rows we can't score
    params = _pick(row, ["#Params (B)", "params_b", "Params (B)"])
    return {
        "model_id": str(name).strip(),
        "display_name": str(name).strip().split("/")[-1],
        "params_b": params,
        "quality": quality,
    }


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
    payload = r.json()
    return [item.get("row", {}) for item in payload.get("rows", [])]


def fetch_leaderboard() -> list[dict[str, Any]]:
    """Pull full HF Open LLM Leaderboard. Returns normalized dicts."""
    out: list[dict[str, Any]] = []
    offset = 0
    while offset < MAX_ROWS:
        try:
            rows = _fetch_page(offset, PAGE_SIZE)
        except requests.RequestException as e:
            log.warning("HF leaderboard fetch failed at offset %d: %s", offset, e)
            break
        if not rows:
            break
        for row in rows:
            norm = _normalize_row(row)
            if norm is not None:
                out.append(norm)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return out
