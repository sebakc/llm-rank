# llm-rank

> Pick the right LLM in one command. Aggregates HuggingFace Open LLM Leaderboard (quality), OpenRouter (cost + context), and your local hardware (VRAM/RAM) into a single ranked recommendation.

No more juggling three browser tabs to figure out whether DeepSeek-Coder is cheaper than GPT-4o, or whether a 14B model fits on your RTX 3060. One CLI, one answer.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why

| Question | Where you'd normally look | With `llm-rank` |
|---|---|---|
| Smartest model under $1/M tokens? | HuggingFace + OpenRouter, by hand | `llm-rank recommend --mode cloud --budget low` |
| What fits my 12 GB GPU? | "Can-it-run-LLM" + leaderboard | `llm-rank recommend --mode local --vram 12gb` |
| Best math model on OpenRouter? | Compare 5 dashboards | `llm-rank recommend --task math --mode cloud` |
| How much VRAM does Llama-3-70B need at Q4? | Wiki + a calculator | `llm-rank info meta-llama/Llama-3-70B` |

---

## Install

Uses [uv](https://github.com/astral-sh/uv) for env + deps.

```bash
git clone https://github.com/yourname/llm-rank ~/projects/skills/llm-rank
cd ~/projects/skills/llm-rank
uv venv
uv pip install -e ".[dev]"
```

Optional extras:

```bash
uv pip install -e ".[gpu]"   # adds torch for CUDA VRAM detection
```

Run via uv (no activation needed):

```bash
uv run llm-rank --version
uv run pytest
```

Or put it on PATH so any shell (and the Claude skill) can find it:

```bash
sudo ln -sf "$PWD/.venv/bin/llm-rank" /usr/local/bin/llm-rank
llm-rank --version
```

> No uv? Fallback: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.

---

## Quick start

```bash
# Cheapest decent coding model on OpenRouter
llm-rank recommend --task coding --mode cloud --budget low

# Best local model that fits 12 GB VRAM at Q5_K_M
llm-rank recommend --task general --mode local --vram 12gb

# Auto-detect host GPU/RAM and recommend a local model
llm-rank hardware
llm-rank recommend --task reasoning --mode local

# Hard $/M tokens cap
llm-rank recommend --task math --mode cloud --max-price 2

# Programmatic / scripting (skills, dashboards, CI)
llm-rank recommend --task coding --mode cloud --json | jq '.recommendations[0]'

# Force-refresh sources (cache TTL is 24h)
llm-rank update
```

### Sample output

```
            llm-rank — task=coding, mode=cloud
┏━━━┳────────────────────────┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳────────────────────────────┓
┃ # ┃ Model                  ┃  Score ┃ Quality┃ $/M tok┃ Notes                      ┃
┡━━━╇────────────────────────╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇────────────────────────────┩
│ 1 │ Meta: Llama 3 8B       │  0.069 │   14.3 │   0.04 │ coding 14.3, $0.04/M, 8k   │
│ 2 │ NousResearch: Hermes 2 │ -0.029 │   22.1 │   0.14 │ coding 22.1, $0.14/M, 8k   │
│ 3 │ Nous: Hermes 3 70B     │ -0.115 │   38.5 │   0.30 │ coding 38.5, $0.30/M, 131k │
└───┴────────────────────────┴────────┴────────┴────────┴────────────────────────────┘
```

---

## Commands

| Command | Purpose |
|---|---|
| `recommend` | Top N models for task + constraints (default 3) |
| `list` | Ranked list without budget/VRAM filtering |
| `info <model-id>` | Show merged record for one model |
| `hardware` | Detected RAM/VRAM |
| `update` | Refresh all caches |

### `recommend` flags

| Flag | Default | Notes |
|---|---|---|
| `--task` | `general` | `coding` \| `general` \| `math` \| `reasoning` |
| `--mode` | `cloud` | `cloud` \| `local` |
| `--budget` | — | `low` ($1/M) \| `med` ($5/M) \| `high` ($50/M) — cloud only |
| `--max-price` | — | Hard $/M-tokens cap; overrides `--budget` |
| `--vram` | auto | e.g. `12gb`; local only. Auto-detected via `torch.cuda` or `nvidia-smi`, else falls back to system RAM |
| `--quant` | `q5_k_m` | `fp16` \| `q8` \| `q5_k_m` \| `q4_k_m` |
| `--top` | `3` | Number of results |
| `--json` | off | Emit JSON to stdout |
| `--refresh` | off | Bypass cache and re-fetch |

---

## Scoring

```
score = quality_norm * w_quality
      - price_norm   * w_cost
      + speed_norm   * w_speed
```

Defaults: `w_quality=1.0`, `w_cost=0.5`, `w_speed=0.3`.

- **Quality** — HF Open LLM Leaderboard v2 columns mapped per task (`Average ⬆️`, `MATH Lvl 5`, `BBH`, …). Falls back to `Average ⬆️` if a task-specific column is absent.
- **Price** — `(prompt + completion) / 2` per 1M tokens, log-scaled across the candidate set. Cheaper is better.
- **Speed** — OpenRouter throughput when available. Neutral when missing.

### Hardware fit

```
required_gb = params_b * bytes_per_param[quant] + 1.5 GB  (KV cache + runtime)
```

| Quant | Bytes / param |
|---|---|
| `fp16` | 2.00 |
| `q8`   | 1.00 |
| `q5_k_m` | 0.70 |
| `q4_k_m` | 0.55 |

> KV-cache cost scales with context length — the 1.5 GB overhead is a rough floor.

---

## Cache

JSON files in `~/.cache/llm-rank/` (override with `LLM_RANK_CACHE_DIR`). 24h TTL. `--refresh` or `llm-rank update` re-fetches.

```bash
ls ~/.cache/llm-rank/
# hf_leaderboard.json  openrouter_models.json
```

---

## Claude Code skill

Install the bundled skill so Claude can call this CLI directly:

```bash
mkdir -p ~/.claude/skills/llm-rank
cp skill/SKILL.md ~/.claude/skills/llm-rank/SKILL.md
```

Then in any Claude Code session, ask things like:

- *"Which open-source coding model can I run on my 8 GB GPU?"*
- *"Cheapest model on OpenRouter that's good at math."*
- *"Compare the top 3 reasoning models under $5/M tokens."*

The skill shells out to `llm-rank --json`, parses the response, and presents the top picks with one-line reasons (plus `ollama run …` when local).

---

## Development

```bash
uv run pytest                   # 13 tests, all unit
uv run ruff check llm_rank tests
uv run llm-rank recommend ...   # manual smoke-test
```

Project layout:

```
llm_rank/
├── cli.py            # Typer entrypoint
├── recommend.py      # orchestrator: merge sources, filter, rank
├── score.py          # pure scoring fns
├── hardware.py       # psutil + torch/nvidia-smi VRAM
├── cache.py          # ~/.cache/llm-rank JSON w/ TTL
├── models.py         # dataclasses
└── sources/
    ├── hf.py         # HF datasets-server REST
    └── openrouter.py # /api/v1/models
tests/
├── fixtures/         # snapshot JSON
└── test_*.py
skill/SKILL.md        # Claude Code skill wrapper
```

---

## Caveats

- The HF v2 leaderboard has no HumanEval column; `--task coding` falls back to the general average. When a coding-specific column lands, `_TASK_COLUMNS` in `sources/hf.py` is the only place to update.
- VRAM fit is a heuristic; very long contexts blow past the 1.5 GB overhead floor.
- OpenRouter prices are mid-market; underlying providers may be cheaper.
- Speed normalization is mostly neutral today — OpenRouter doesn't expose throughput consistently.

---

## Roadmap (post-v1)

- Live latency probes against OpenRouter
- LMSYS Arena + MTEB fusion
- Emit `ollama pull` / `litellm` config blocks
- `llm-rank serve` HTTP wrapper
- MCP server mode for direct tool-use integration

---

## License

MIT.
