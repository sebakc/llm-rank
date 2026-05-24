# Plan: `llm-rank` — LLM Decision CLI + Claude Skill

## Context

No single tool aggregates LLM quality (HuggingFace leaderboard), cost/speed (OpenRouter), and hardware fit (local VRAM) into one recommendation. Developers manually cross-reference 3 sites. This project builds a Python CLI (`llm-rank`) that fetches all three, scores models against user constraints, and emits a ranked recommendation. A thin Claude Code skill wraps the CLI so Claude can invoke it directly.

**Working dir**: `/root/projects/skills/llm-rank/` (currently empty — greenfield).

**Decisions locked** (from clarification):
- Language: **Python 3.10+**
- Skill integration: **Skill wraps CLI** (shell-out, no MCP)
- Data: **24h cache + `--refresh`** in `~/.cache/llm-rank/`
- Scope v1: **Cloud + local both**

---

## Architecture

```
llm-rank/
├── pyproject.toml              # package + [project.scripts] llm-rank = "llm_rank.cli:app"
├── README.md
├── llm_rank/
│   ├── __init__.py
│   ├── cli.py                  # Typer entrypoint; subcommands
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── hf.py               # HF Open LLM Leaderboard via `datasets`
│   │   └── openrouter.py       # GET https://openrouter.ai/api/v1/models
│   ├── hardware.py             # psutil RAM + torch.cuda VRAM detection
│   ├── cache.py                # JSON cache w/ TTL in ~/.cache/llm-rank/
│   ├── score.py                # weighted scoring: quality, price, speed, fit
│   ├── recommend.py            # orchestrator: merge sources, filter, rank
│   └── models.py               # dataclasses: ModelEntry, Constraints, Recommendation
└── tests/
    ├── test_score.py
    ├── test_cache.py
    └── test_recommend.py       # uses fixture JSON
```

### Data flow

```
[HF leaderboard] ─┐
                  ├─> merge by canonical model name ─> filter (mode/vram/budget) ─> score ─> rank ─> top N
[OpenRouter API] ─┤
                  │
[Local hardware] ─┘ (only when --mode local)
```

### Scoring

```python
# llm_rank/score.py
score = (quality_norm * w_quality) - (price_norm * w_cost) + (speed_norm * w_speed)
# weights default: quality=1.0, cost=0.5, speed=0.3 — tunable via flags
```

`quality_norm` from HF `average` column (0–100 → 0–1). `price_norm` from OpenRouter `pricing.prompt + pricing.completion` per 1M tokens (log-scaled). `speed_norm` from OpenRouter throughput when available.

### Hardware-fit rule

```
required_vram_gb ≈ params_billion * bytes_per_param
  fp16 → 2 bytes; q8 → 1; q5_k_m → 0.7; q4_k_m → 0.55
```

`--vram 12gb` filters out anything > 12 GB at the chosen quant (default q5_k_m for local mode). When user omits `--vram`, detect via `torch.cuda.get_device_properties(0).total_memory` and fall back to `psutil.virtual_memory().total` for CPU-only.

---

## CLI surface

```
llm-rank recommend --task <coding|general|math|reasoning> \
                   --mode <cloud|local> \
                   [--budget low|med|high]  # cloud only
                   [--max-price 5.0]         # $/M tokens
                   [--vram 12gb]             # local only; auto-detect if omitted
                   [--quant q4_k_m|q5_k_m|q8|fp16]
                   [--top 3]
                   [--json]
                   [--refresh]

llm-rank list   [--mode ...] [--task ...] [--top 20]
llm-rank info <model-id>
llm-rank hardware                  # show detected GPU/RAM
llm-rank update                    # force-refresh all caches
llm-rank --version
```

Task → HF column map (in `recommend.py`):
- `coding` → `humaneval` / `mbpp` (fallback `average`)
- `math` → `gsm8k`
- `reasoning` → `arc_challenge` / `bbh`
- `general` → `average`

Output (default human): Rich table — rank, model, score, quality, $/M, fit. `--json` for skill/programmatic use.

---

## Claude Skill

Path: `~/.claude/skills/llm-rank/SKILL.md`

```markdown
---
name: llm-rank
description: Recommend the best LLM for a task given budget, hardware, or speed constraints. Aggregates HuggingFace leaderboard quality, OpenRouter cloud pricing, and local VRAM fit. Trigger when user asks "which model should I use", "best LLM for X", "what runs on my GPU", "cheapest model for Y".
---

# llm-rank

Shell out to `llm-rank` CLI. Always pass `--json` so output parses cleanly.

## Usage

- Cloud pick: `llm-rank recommend --task coding --mode cloud --budget low --json`
- Local pick (auto-detect VRAM): `llm-rank recommend --task general --mode local --json`
- Local pick (explicit): `llm-rank recommend --task reasoning --mode local --vram 12gb --quant q5_k_m --json`
- Refresh stale data: append `--refresh`

Parse the JSON, present the top 1–3 with a one-line reason each. If the user wants to run locally, suffix with `ollama run <model-id>` when an Ollama tag exists.
```

---

## Key files to create

| File | Purpose |
|------|---------|
| `pyproject.toml` | Build config, deps (`typer`, `rich`, `requests`, `datasets`, `psutil`, `pydantic`; optional `torch`), entrypoint |
| `llm_rank/cli.py` | Typer app, subcommands wire into `recommend.py` / `hardware.py` |
| `llm_rank/sources/hf.py` | `load_dataset("open-llm-leaderboard/contents")` → normalized list |
| `llm_rank/sources/openrouter.py` | `requests.get("https://openrouter.ai/api/v1/models")` → normalized list |
| `llm_rank/cache.py` | `get(key, ttl_h, loader)` reads/writes `~/.cache/llm-rank/<key>.json` w/ mtime check |
| `llm_rank/score.py` | Pure functions: `normalize()`, `score_entry()`, `rank()` |
| `llm_rank/recommend.py` | Merge HF + OpenRouter by name heuristic; apply filters; call `score.rank()` |
| `llm_rank/hardware.py` | `detect_vram_gb()`, `detect_ram_gb()`, `fits(model_params_b, quant, available_gb)` |
| `tests/test_*.py` | pytest; mock HTTP via fixture JSON in `tests/fixtures/` |
| `~/.claude/skills/llm-rank/SKILL.md` | Skill wrapper |

---

## Install / dev

```bash
cd /root/projects/skills/llm-rank
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # editable install → `llm-rank` on PATH
pytest
llm-rank --version
```

Optional `[gpu]` extra for `torch` so hardware detect works without forcing torch on cloud-only users.

---

## Verification

1. **Unit**: `pytest` — score math, cache TTL, hardware fit rule against fixture data.
2. **E2E cloud**: `llm-rank recommend --task coding --mode cloud --budget low` → returns ranked table including a known cheap coder (e.g. `deepseek-coder`).
3. **E2E local**: `llm-rank hardware` shows real GPU/RAM; `llm-rank recommend --task general --mode local --vram 8gb` returns models ≤ 8 GB at default quant.
4. **Cache**: first call slow, second call < 100 ms, `--refresh` re-hits network.
5. **Skill**: in a Claude Code session, ask "best free coding model on OpenRouter" — skill triggers, CLI runs, top pick reported.
6. **JSON contract**: `llm-rank recommend ... --json | jq '.recommendations[0].model_id'` returns non-null.

---

## Out of scope (v2+)

- Live throughput benchmarks (latency probing against OpenRouter)
- Multi-leaderboard fusion (LMSYS Arena, MTEB)
- Auto-emit `ollama pull` / `litellm` config
- Web UI / `llm-rank serve`
