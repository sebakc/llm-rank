"""Orchestrator: load sources, merge, filter, rank, build recommendations."""

from __future__ import annotations

import logging
import re
from typing import Any

from . import cache, hardware, score
from .models import Constraints, ModelEntry, Recommendation
from .sources import arena, hf, openrouter, openrouter_frontend

log = logging.getLogger(__name__)

CACHE_TTL_HOURS = 24.0
HF_CACHE_KEY = "hf_leaderboard"
OR_CACHE_KEY = "openrouter_models"
ORF_CACHE_KEY = "openrouter_frontend"
ARENA_CACHE_KEY = "arena_leaderboard"
# Frontend perf data churns faster; refresh more aggressively.
ORF_TTL_HOURS = 6.0

# Budget cap in $/M tokens (price_avg)
BUDGET_CAPS: dict[str, float] = {
    "low": 1.0,
    "med": 5.0,
    "high": 50.0,
}


def _slug(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _squash(name: str) -> str:
    """Aggressive normalization: lowercase, drop all non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


_VERSION_TAIL = re.compile(r"v\d+(?:\.\d+)*$")


_TRAILING_DATE = re.compile(r"(?:20\d{6}|[-_ ]?\d{4,8})$")
_PAREN = re.compile(r"\s*\([^)]*\)")


def _canonical_keys(model_id: str) -> set[str]:
    """Generate several lookup keys so HF / OpenRouter / Arena ids merge."""
    raw = model_id.lower().strip()
    # strip parenthetical qualifiers ("(thinking-32k)", "(20250514)")
    raw_no_paren = _PAREN.sub("", raw).strip()
    # strip openrouter ":free", ":nitro", etc.
    base = raw_no_paren.split(":", 1)[0]
    short = base.split("/", 1)[-1]
    keys = {raw, raw_no_paren, base, short}
    keys.add(_slug(base))
    keys.add(_slug(short))
    sq = _squash(short)
    keys.add(sq)
    # squash without trailing version tag
    keys.add(_VERSION_TAIL.sub("", sq))
    # squash without trailing date suffix (e.g. grok-4-0709 -> grok4)
    keys.add(_TRAILING_DATE.sub("", sq))
    # Common provider-prefix variants Arena uses ("ChatGPT-4o" vs "gpt-4o").
    if sq.startswith("chatgpt"):
        keys.add(sq.replace("chatgpt", "gpt", 1))
    keys = {k for k in keys if k and len(k) >= 3}
    return keys


def load_hf(refresh: bool = False) -> list[dict[str, Any]]:
    return cache.get(HF_CACHE_KEY, hf.fetch_leaderboard, CACHE_TTL_HOURS, refresh)


def load_openrouter(refresh: bool = False) -> list[dict[str, Any]]:
    return cache.get(OR_CACHE_KEY, openrouter.fetch_models, CACHE_TTL_HOURS, refresh)


def load_arena(refresh: bool = False) -> list[dict[str, Any]]:
    return cache.get(ARENA_CACHE_KEY, arena.fetch_leaderboard, CACHE_TTL_HOURS, refresh)


def load_openrouter_frontend(refresh: bool = False) -> list[dict[str, Any]]:
    return cache.get(
        ORF_CACHE_KEY,
        lambda: openrouter_frontend.fetch_all(progress=True),
        ORF_TTL_HOURS,
        refresh,
    )


def merge(
    hf_rows: list[dict[str, Any]],
    or_rows: list[dict[str, Any]],
    arena_rows: list[dict[str, Any]] | None = None,
) -> list[ModelEntry]:
    """Merge HF leaderboard quality + LMArena Elo + OpenRouter cloud rows.

    `or_rows` is the unified OR record (preferably frontend-source with perf data;
    falls back to public /api/v1/models shape). `arena_rows` is the LMArena
    Elo leaderboard; used as fallback quality for closed/frontier models that
    HF doesn't evaluate.
    """
    arena_rows = arena_rows or []
    # Index HF by every canonical key it generates.
    hf_index: dict[str, dict[str, Any]] = {}
    for row in hf_rows:
        for k in _canonical_keys(row["model_id"]):
            hf_index.setdefault(k, row)

    arena_index: dict[str, dict[str, Any]] = {}
    for row in arena_rows:
        for k in _canonical_keys(row["model_id"]):
            arena_index.setdefault(k, row)

    entries: list[ModelEntry] = []
    used_hf_ids: set[str] = set()

    for orm in or_rows:
        keys = _canonical_keys(orm["model_id"])
        # Frontend rows carry hf_slug directly; use it first.
        if orm.get("hf_slug"):
            keys |= _canonical_keys(orm["hf_slug"])
        hf_match = next((hf_index[k] for k in keys if k in hf_index), None)
        arena_match = next((arena_index[k] for k in keys if k in arena_index), None)
        # Frontend rows expose `price_in_per_m`; public rows use `price_in`.
        price_in = orm.get("price_in_per_m", orm.get("price_in"))
        price_out = orm.get("price_out_per_m", orm.get("price_out"))
        entry = ModelEntry(
            model_id=orm["model_id"],
            display_name=orm.get("display_name", orm["model_id"]),
            price_in=price_in,
            price_out=price_out,
            context_len=orm.get("context_len"),
            throughput_tps=orm.get("throughput_p50"),
            latency_ms=orm.get("latency_p50_ms"),
            uptime_pct=orm.get("uptime_pct"),
            providers=orm.get("providers", []),
            provider_pricing=orm.get("provider_pricing", []),
            best_provider=orm.get("best_provider"),
            provider="openrouter",
            available_cloud=True,
        )
        if hf_match is not None:
            entry.quality = dict(hf_match.get("quality", {}))
            entry.params_b = hf_match.get("params_b")
            entry.available_local = True
            used_hf_ids.add(hf_match["model_id"])
        elif arena_match is not None:
            # Closed/frontier model with Arena Elo but no HF eval.
            entry.quality = dict(arena_match.get("quality", {}))
        entries.append(entry)

    # HF-only models (open weights not on OpenRouter -> local-only)
    for row in hf_rows:
        if row["model_id"] in used_hf_ids:
            continue
        entries.append(
            ModelEntry(
                model_id=row["model_id"],
                display_name=row.get("display_name", row["model_id"]),
                params_b=row.get("params_b"),
                quality=dict(row.get("quality", {})),
                available_local=True,
            )
        )

    return entries


def _apply_filters(
    entries: list[ModelEntry],
    c: Constraints,
    hw: hardware.HardwareInfo | None,
    rated_only: bool = False,
) -> list[ModelEntry]:
    out: list[ModelEntry] = []
    for e in entries:
        if c.mode == "cloud":
            if not e.available_cloud:
                continue
            cap = c.max_price
            if cap is None and c.budget is not None:
                cap = BUDGET_CAPS.get(c.budget)
            if cap is not None and e.price_avg is not None and e.price_avg > cap:
                continue
        else:  # local
            if not e.available_local:
                continue
            if e.params_b is None:
                continue
            avail = c.vram_gb
            if avail is None and hw is not None:
                avail = hw.vram_gb if hw.vram_gb else hw.ram_gb
            if avail is not None and not hardware.fits(e.params_b, c.quant, avail):
                continue
            # Local mode always requires a quality signal (no point ranking blind).
            if score.quality_for(e, c.task) is None:
                continue
        if rated_only and score.quality_for(e, c.task) is None:
            continue
        out.append(e)
    return out


def _reason(entry: ModelEntry, c: Constraints) -> str:
    parts: list[str] = []
    q = score.quality_for(entry, c.task)
    if q is not None:
        parts.append(f"{c.task} {q:.1f}")
    if c.mode == "cloud" and entry.price_avg is not None:
        parts.append(f"${entry.price_avg:.2f}/M")
    if c.mode == "local" and entry.params_b:
        req = hardware.required_gb(entry.params_b, c.quant)
        parts.append(f"~{req:.1f} GB @ {c.quant}")
    if entry.throughput_tps:
        parts.append(f"{entry.throughput_tps:.0f} t/s")
    if entry.latency_ms:
        parts.append(f"{entry.latency_ms / 1000:.1f}s TTFT")
    if entry.context_len:
        parts.append(f"{entry.context_len // 1000}k ctx")
    if entry.best_provider:
        parts.append(f"via {entry.best_provider}")
    return ", ".join(parts) if parts else "ranked by score"


def recommend(c: Constraints, refresh: bool = False, rated_only: bool = False) -> list[Recommendation]:
    hf_rows = load_hf(refresh=refresh)
    arena_rows = load_arena(refresh=refresh)
    # Prefer the frontend source (real perf + larger catalog). Fall back to
    # the public /api/v1/models endpoint if it returns nothing.
    or_rows = load_openrouter_frontend(refresh=refresh)
    if not or_rows:
        or_rows = load_openrouter(refresh=refresh)
    entries = merge(hf_rows, or_rows, arena_rows)
    hw = hardware.detect() if c.mode == "local" else None
    filtered = _apply_filters(entries, c, hw, rated_only=rated_only)
    ranked = score.rank(filtered, c)[: c.top]
    recs: list[Recommendation] = []
    for i, (e, s) in enumerate(ranked, start=1):
        fits_gb = None
        if e.params_b is not None:
            fits_gb = hardware.required_gb(e.params_b, c.quant)
        ollama = None
        if c.mode == "local":
            # heuristic: derive tag from short name
            short = e.model_id.split("/", 1)[-1].lower()
            ollama = f"ollama run {short}"
        recs.append(
            Recommendation(
                rank=i,
                model_id=e.model_id,
                display_name=e.display_name,
                score=round(s, 4),
                quality=score.quality_for(e, c.task),
                price_avg=e.price_avg,
                fits_vram_gb=fits_gb,
                reason=_reason(e, c),
                ollama_cmd=ollama,
                throughput_tps=e.throughput_tps,
                latency_ms=e.latency_ms,
                uptime_pct=e.uptime_pct,
                best_provider=e.best_provider,
                providers=list(e.providers),
                provider_pricing=list(e.provider_pricing),
                context_len=e.context_len,
            )
        )
    return recs
