"""Core dataclasses shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


Task = Literal["coding", "general", "math", "reasoning"]
Mode = Literal["cloud", "local"]
Budget = Literal["low", "med", "high"]
Quant = Literal["fp16", "q8", "q5_k_m", "q4_k_m"]


@dataclass
class ModelEntry:
    """Unified model record after merging HF + OpenRouter."""

    model_id: str                       # canonical id, e.g. "deepseek-ai/deepseek-coder-v2"
    display_name: str
    params_b: Optional[float] = None    # billions
    quality: dict = field(default_factory=dict)  # task -> 0..100
    price_in: Optional[float] = None    # $/M prompt tokens
    price_out: Optional[float] = None   # $/M completion tokens
    context_len: Optional[int] = None
    throughput_tps: Optional[float] = None   # best provider p50 tok/s
    latency_ms: Optional[float] = None       # best provider p50 TTFT ms
    uptime_pct: Optional[float] = None       # last-30m uptime %
    providers: list = field(default_factory=list)
    provider_pricing: list = field(default_factory=list)
    best_provider: Optional[str] = None
    supports_reasoning: bool = False
    knowledge_cutoff: Optional[str] = None
    provider: Optional[str] = None      # "openrouter" route id
    available_cloud: bool = False
    available_local: bool = False       # heuristic: open weights present
    ollama_tag: Optional[str] = None

    @property
    def price_avg(self) -> Optional[float]:
        if self.price_in is None and self.price_out is None:
            return None
        a = self.price_in or 0.0
        b = self.price_out or 0.0
        return (a + b) / 2.0


@dataclass
class Constraints:
    task: Task = "general"
    mode: Mode = "cloud"
    budget: Optional[Budget] = None
    max_price: Optional[float] = None       # $/M tokens cap
    vram_gb: Optional[float] = None         # explicit; auto-detect if None
    quant: Quant = "q5_k_m"
    top: int = 3
    weights: dict = field(default_factory=lambda: {"quality": 1.0, "cost": 0.5, "speed": 0.3})


@dataclass
class Recommendation:
    rank: int
    model_id: str
    display_name: str
    score: float
    quality: Optional[float]
    price_avg: Optional[float]
    fits_vram_gb: Optional[float]
    reason: str
    ollama_cmd: Optional[str] = None
    throughput_tps: Optional[float] = None
    latency_ms: Optional[float] = None
    uptime_pct: Optional[float] = None
    best_provider: Optional[str] = None
    providers: list = field(default_factory=list)
    provider_pricing: list = field(default_factory=list)
    context_len: Optional[int] = None
    supports_reasoning: bool = False
    knowledge_cutoff: Optional[str] = None
