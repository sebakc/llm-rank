"""User configuration: budget bands, custom overrides.

Bands are price ranges per tier (low/med/high) in $/M tokens (avg of in+out).
Defaults are chosen from the actual OpenRouter catalog distribution:
- low  : $0.00 – $1.00  (covers ~p50 of catalog: cheap workhorses)
- med  : $1.00 – $10.00 (covers p50–p90: premium)
- high : $10.00 – ∞    (covers p90+: frontier)

Overrides live in $LLM_RANK_CONFIG (default ~/.config/llm-rank/config.toml):

    [bands]
    low  = [0, 0.5]
    med  = [0.5, 5]
    high = [5, "inf"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


CONFIG_PATH = Path(
    os.environ.get("LLM_RANK_CONFIG", str(Path.home() / ".config" / "llm-rank" / "config.toml"))
)


# (min_inclusive, max_exclusive). None on max = open upper end.
DEFAULT_BANDS: dict[str, tuple[float, Optional[float]]] = {
    "low":  (0.0, 1.0),
    "med":  (1.0, 10.0),
    "high": (10.0, None),
}


def _coerce_band(value) -> tuple[float, Optional[float]]:
    """Accept [a, b] or [a, "inf"] from TOML."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"band must be a 2-element list, got: {value!r}")
    lo, hi = value
    lo = float(lo)
    if hi is None or (isinstance(hi, str) and hi.lower() in ("inf", "infinity", "")):
        return (lo, None)
    return (lo, float(hi))


def load_bands() -> dict[str, tuple[float, Optional[float]]]:
    """Return active bands. Defaults overridden by user config file if present."""
    bands = dict(DEFAULT_BANDS)
    if not CONFIG_PATH.exists():
        return bands
    try:
        data = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return bands
    overrides = data.get("bands") or {}
    for tier, raw in overrides.items():
        if tier not in bands:
            continue
        try:
            bands[tier] = _coerce_band(raw)
        except (TypeError, ValueError):
            continue
    return bands


def write_default_config() -> Path:
    """Materialize a starter config the user can edit."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    body = (
        "# llm-rank config\n"
        "# Bands are [min, max] in $/M tokens (avg of input + output).\n"
        "# Use \"inf\" for an open upper end.\n\n"
        "[bands]\n"
        "low  = [0, 1]\n"
        "med  = [1, 10]\n"
        "high = [10, \"inf\"]\n"
    )
    CONFIG_PATH.write_text(body, encoding="utf-8")
    return CONFIG_PATH


def band_label(band: tuple[float, Optional[float]]) -> str:
    lo, hi = band
    hi_s = "∞" if hi is None else f"${hi:.2f}"
    return f"${lo:.2f}–{hi_s}/M"
