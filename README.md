# llm-rank

> Pick the right LLM in one command. Aggregates real-time OpenRouter perf + pricing, HuggingFace Open LLM Leaderboard, and LMArena Elo into one ranked recommendation.

No more juggling browser tabs comparing model quality, prices, throughput, and provider routes. One CLI, one answer.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why

| Question | Without `llm-rank` | With `llm-rank` |
|---|---|---|
| Smartest model under $1/M? | Cross-ref 3 dashboards | `llm-rank list --budget low` |
| What fits my 12 GB GPU? | "Can-it-run-LLM" + leaderboard | `llm-rank recommend --mode local --vram 12gb` |
| Best Claude provider on OpenRouter? | Click each provider page | `llm-rank list -f anth --top 5` |
| GPT-5 vs DeepSeek V4 vs Llama 405B — apples to apples? | Bench + price hunt | `llm-rank list -d --top 10` |
| Cheap reasoning model with fast TTFT? | Filter manually | `llm-rank list --max-price 1 --min-quality 70` |

---

## Install

Uses [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/sebakc/llm-rank
cd llm-rank
uv tool install .
llm-rank --version
```

`uv tool install` creates an isolated env and puts `llm-rank` on your PATH — no `source .venv/bin/activate` ever.

GPU extras (local VRAM detection via torch):

```bash
uv tool install ".[gpu]"
```

Upgrade / remove / self-update:

```bash
uv tool upgrade llm-rank
llm-rank self-update              # pulls latest from GitHub
llm-rank self-update --check      # version compare only
uv tool uninstall llm-rank
```

Background version check runs on every invocation (cached 24h). Disable with `--no-update-check`.

### Dev install (editable, for hacking)

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
uv run llm-rank ...
```

> No uv? Fallback: `pipx install .`

---

## Quick start

```bash
# Top 50 cloud models, ranked overall
llm-rank list

# Cheap coding models (band-filtered)
llm-rank list --task coding --budget low

# Detailed view: reasoning badge, cutoff, quantization, cache price, per-provider $in/$out
llm-rank list --detailed -f deepseek --top 5

# Best local pick for 12 GB GPU at Q5_K_M
llm-rank recommend --mode local --vram 12gb --task general

# Hard $/M cap with quality floor
llm-rank list --max-price 0.5 --min-quality 70

# Substring filter (matches id + display name)
llm-rank list -f openai           # all OpenAI models
llm-rank list -f anth -n 3        # top 3 Anthropic

# Programmatic / scripting
llm-rank list -f gpt-5 --format json | jq '.recommendations[0]'
llm-rank list --all --format csv > catalog.csv

# Refresh sources (cache TTL: 6h for OR perf, 24h for HF + Arena)
llm-rank update
```

---

## Commands

| Command | Purpose |
|---|---|
| `list` | Ranked catalog with filters (top N, budget, price, quality, substring) |
| `recommend` | Single decision: top N for task + constraints + optional VRAM check |
| `info <model-id>` | Show merged record for one model |
| `hardware` | Detected RAM/VRAM |
| `bands` | Show price bands per budget tier + catalog distribution |
| `bands --init` | Write starter config at `~/.config/llm-rank/config.toml` |
| `update` | Refresh all caches |
| `self-update` | Upgrade llm-rank itself |

### Shared flags (`list` and `recommend`)

| Flag | Default | Notes |
|---|---|---|
| `--task` | `general` | `coding` \| `general` \| `math` \| `reasoning` |
| `--mode` | `cloud` | `cloud` \| `local` |
| `--top N` / `-n N` | 50 (list), 3 (recommend) | `--top 0` or `--all` = unlimited |
| `--budget` | — | `low` \| `med` \| `high` — applies price band + cost weight |
| `--max-price` | — | Hard $/M cap (overrides budget cap) |
| `--min-quality` | — | 0..100 floor |
| `--filter` / `-f` | — | Substring match on id or display name |
| `--detailed` / `-d` | off | Show extra cols: 🧠 reasoning badge, cutoff date, quantization, cache price |
| `--format` | `table` | `table` \| `csv` \| `json` |
| `--refresh` | off | Bypass cache and re-fetch |
| `--rated-only` | off | Hide models without HF/Arena quality data |
| `--quant` | `q5_k_m` | Local mode: `fp16` \| `q8` \| `q5_k_m` \| `q4_k_m` |
| `--vram` | auto | Local mode: e.g. `12gb` — auto-detected from torch/nvidia-smi if omitted |

---

## Price bands (`--budget`)

Bands are **mutually exclusive** — `--budget med` won't show $0.10/M cheap models even if they'd score well. Defaults derived from the actual OpenRouter catalog distribution:

| Tier | Band | Catalog share | Cost weight |
|---|---|---|---|
| `low` | $0–$1 / M | ~50% | 1.5 (cheap wins) |
| `med` | $1–$10 / M | ~40% | 0.5 (balanced) |
| `high` | $10+ / M | ~10% | 0.15 (quality wins) |

Override via `~/.config/llm-rank/config.toml`:

```toml
[bands]
low  = [0, 0.5]
med  = [0.5, 3]
high = [3, "inf"]
```

Env var override: `LLM_RANK_CONFIG=/path/to/your.toml`.

---

## Data sources

| Source | What we use | TTL |
|---|---|---|
| **OpenRouter frontend** (`/api/frontend/*`) | Catalog (754 models), per-provider p50/p95 throughput + latency, uptime, request count, pricing (incl. cache read), quantization, knowledge cutoff, reasoning capability | 6h |
| **HuggingFace Open LLM Leaderboard v2** | Quality for open-weight models (BBH, MATH, GPQA, MUSR, MMLU-PRO, Average) | 24h |
| **LMArena Elo** (`mathewhe/chatbot-arena-elo` HF dataset) | Quality for closed/frontier models (Claude, GPT, Gemini, Grok, o-series) via Arena Score | 24h |

OR's public `/api/v1/models` is used only as a fallback when the frontend endpoint is down.

---

## Scoring

```
score = quality_norm * w_quality - price_norm * w_cost + speed_norm * w_speed
```

Defaults: `quality=1.0`, `cost=0.5`, `speed=0.3`. `--budget` overrides `cost` weight per tier (above).

- **Quality**: HF v2 score (per task) when available; falls back to LMArena Elo normalized to 0–100; unrated models grouped at the bottom and sorted by price.
- **Price**: `(in + out) / 2` per 1M tokens (cheapest provider), log-scaled across the candidate set.
- **Speed**: best provider's p50 throughput (tok/s), normalized to candidate-set max.

### Hardware fit (local mode)

```
required_gb = params_b * bytes_per_param[quant] + 1.5 GB overhead
```

| Quant | Bytes/param |
|---|---|
| `fp16` | 2.00 |
| `q8` | 1.00 |
| `q5_k_m` | 0.70 |
| `q4_k_m` | 0.55 |

---

## Output formats

**Table** (default) — color-coded, with quality bars, reasoning badges, per-provider pricing per row.

**JSON** — full record per recommendation including `provider_pricing` array.

```bash
llm-rank list -f gpt-5 --format json
```

**CSV** — every Recommendation field, ready for spreadsheets.

```bash
llm-rank list --all --format csv > catalog.csv
```

---

## Cache

JSON files in `~/.cache/llm-rank/` (override with `LLM_RANK_CACHE_DIR`). TTLs vary by source. `--refresh` or `llm-rank update` re-fetches everything.

```bash
ls ~/.cache/llm-rank/
# hf_leaderboard.json  arena_leaderboard.json  openrouter_frontend.json  openrouter_models.json  self_version_check.json
```

---

## Claude Code skill

Install the bundled skill so Claude can call this CLI directly:

```bash
mkdir -p ~/.claude/skills/llm-rank
cp skill/SKILL.md ~/.claude/skills/llm-rank/SKILL.md
```

Then ask Claude:

- *"Cheapest reasoning model on OpenRouter."*
- *"Best Anthropic model under $5/M."*
- *"Which open-source coder fits my 12 GB GPU?"*

Skill shells out to `llm-rank --json`, parses the result, and presents top picks.

---

## Development

```bash
uv run pytest                   # 13 tests
uv run ruff check llm_rank tests
uv run llm-rank recommend ...
```

Project layout:

```
llm_rank/
├── cli.py                  # Typer entrypoint
├── recommend.py            # orchestrator: merge sources, filter, rank
├── score.py                # pure scoring fns
├── hardware.py             # psutil + torch/nvidia-smi VRAM
├── cache.py                # ~/.cache/llm-rank JSON w/ TTL
├── config.py               # ~/.config/llm-rank/config.toml bands
├── updater.py              # self-update + version check
├── models.py               # dataclasses (ModelEntry, Constraints, Recommendation)
└── sources/
    ├── openrouter_frontend.py  # /api/frontend/* (primary cloud source)
    ├── openrouter.py           # /api/v1/models (fallback)
    ├── arena.py                # LMArena Elo via HF dataset
    └── hf.py                   # Open LLM Leaderboard v2 via HF datasets-server
tests/
├── fixtures/               # snapshot JSON
└── test_*.py
skill/SKILL.md              # Claude Code skill wrapper
```

---

## Caveats

- OR frontend API is **undocumented** — could break. Code degrades gracefully (returns empty) and falls back to public `/api/v1/models`.
- HF v2 leaderboard misses frontier/proprietary models; Arena fills that gap, but its scores cover the whole task set (no per-task split).
- VRAM fit is heuristic; long contexts blow past the 1.5 GB overhead floor.
- Cheapest-provider pricing ≠ OR's traffic-weighted "effective" price (cache hit rate not exposed publicly).
- Speed shown is best provider's p50 throughput; tail (p95/p99) available in JSON output.

---

## Roadmap (post-v0.2)

- Active latency probes (real RTT measurement)
- Cache-adjusted effective pricing (requires hit-rate signal)
- Multi-leaderboard fusion (MTEB, Aider, SWE-bench, SimpleBench)
- Auto-emit `ollama pull` / `litellm config` blocks
- `llm-rank serve` HTTP wrapper
- MCP server mode

---

## License

MIT.
