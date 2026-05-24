---
name: llm-rank
description: Recommend the best LLM for a given task and constraints. Aggregates HuggingFace Open LLM Leaderboard quality, OpenRouter cloud pricing, and local VRAM fit. Trigger when the user asks "which model should I use", "best LLM for X", "what model fits my GPU", "cheapest model for Y", "compare LLMs", or otherwise needs a model recommendation.
---

# llm-rank

Use this skill when the user wants help picking an LLM. Shell out to the `llm-rank` CLI; always pass `--json` so the output is structured.

## Pick the right invocation

Map the user's intent to flags:

| Intent | Command |
|---|---|
| Best cheap cloud model | `llm-rank recommend --task <T> --mode cloud --budget low --json` |
| Hard $/M cap | `llm-rank recommend --task <T> --mode cloud --max-price <N> --json` |
| Best local model, auto-detect VRAM | `llm-rank recommend --task <T> --mode local --json` |
| Local with known GPU | `llm-rank recommend --task <T> --mode local --vram <N>gb --quant q5_k_m --json` |
| Just show host hardware | `llm-rank hardware` |
| Lookup a single model | `llm-rank info <model-id>` |
| Force-refresh sources | append `--refresh` (cache TTL is 24h) |

`<T>` is one of `coding`, `general`, `math`, `reasoning`.

Use `--top 5` if the user wants alternatives, otherwise default `--top 3` is fine.

## Presenting results

1. Parse the JSON from stdout. The shape is `{ "constraints": {...}, "recommendations": [ {rank, model_id, display_name, score, quality, price_avg, fits_vram_gb, reason, ollama_cmd}, ... ] }`.
2. Lead with the rank-1 model, one sentence on *why* (use the `reason` field), then list the next 1–2 alternatives compactly.
3. If `mode=local` and `ollama_cmd` is populated, append it as a copy-paste-ready next step.
4. If `recommendations` is empty, tell the user no model matched and suggest loosening a constraint (raise `--max-price`, drop `--budget`, raise `--vram`, switch `--quant` to `q4_k_m`).

## Caveats

- Quality scores come from HF Open LLM Leaderboard v2; the `coding` task falls back to the general average when no coding-specific column is present.
- VRAM fit uses a heuristic (`params_b * bytes_per_quant + 1.5 GB overhead`); real KV-cache usage scales with context length.
- OpenRouter prices are mid-market; some routes may be cheaper at the underlying provider.
