"""Typer CLI entrypoint for `llm-rank`."""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from dataclasses import asdict, fields
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__, cache, hardware, recommend
from .models import Constraints, Recommendation

VALID_FORMATS = ("table", "csv", "json")


def _emit_csv(recs: list[Recommendation]) -> None:
    if not recs:
        return
    cols = [f.name for f in fields(Recommendation)]
    w = csv.writer(sys.stdout)
    w.writerow(cols)
    for r in recs:
        row = []
        for c in cols:
            v = getattr(r, c)
            if isinstance(v, (list, dict)):
                v = json.dumps(v)
            row.append("" if v is None else v)
        w.writerow(row)


def _emit_json(recs: list[Recommendation], c: Constraints) -> None:
    payload = {
        "constraints": {
            "task": c.task,
            "mode": c.mode,
            "budget": c.budget,
            "max_price": c.max_price,
            "vram_gb": c.vram_gb,
            "quant": c.quant,
            "top": c.top,
        },
        "recommendations": [asdict(r) for r in recs],
    }
    typer.echo(json.dumps(payload, indent=2))

app = typer.Typer(
    name="llm-rank",
    help="Rank and recommend LLMs by quality, cost, speed, and hardware fit.",
    no_args_is_help=True,
)
console = Console()


def _parse_size_gb(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(gb|g|gib)?\s*$", value, re.IGNORECASE)
    if not m:
        raise typer.BadParameter(f"Cannot parse size: {value!r}. Try '12gb'.")
    return float(m.group(1))


def _quality_bar(q: float | None, width: int = 8) -> Text:
    if q is None:
        return Text("—", style="dim")
    pct = max(0.0, min(1.0, q / 100.0))
    filled = int(round(pct * width))
    bar = "█" * filled + "░" * (width - filled)
    if pct >= 0.7:
        color = "green"
    elif pct >= 0.4:
        color = "yellow"
    else:
        color = "red"
    return Text.assemble((bar, color), "  ", (f"{q:5.1f}", "bold"))


def _score_text(score: float) -> Text:
    if score >= 0.3:
        style = "bold green"
    elif score >= 0:
        style = "yellow"
    else:
        style = "red"
    return Text(f"{score:+.3f}", style=style)


def _price_text(price: float | None) -> Text:
    if price is None:
        return Text("—", style="dim")
    if price < 0.5:
        style = "green"
    elif price < 5:
        style = "yellow"
    else:
        style = "red"
    return Text(f"${price:6.2f}", style=style)


def _trunc(s: str, n: int = 38) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _providers_text(provs: list[str] | None, best: str | None = None) -> str:
    if not provs:
        return "—"
    seen: dict[str, None] = {}
    for p in provs:
        if p:
            seen.setdefault(p, None)
    uniq = list(seen.keys())
    head = best if best in uniq else uniq[0]
    extra = len(uniq) - 1
    return f"{head}" + (f" ×{len(uniq)}" if extra else "")


def _provider_pricing_text(pp: list[dict] | None, limit: int = 6) -> Text:
    """One line per provider: 'Name  $in/$out'. Cheapest first."""
    if not pp:
        return Text("—", style="dim")
    lines = Text()
    for i, p in enumerate(pp[:limit]):
        if i:
            lines.append("\n")
        name = p.get("provider") or "?"
        pi = p.get("price_in_per_m")
        po = p.get("price_out_per_m")
        if pi is None and po is None:
            price_part = "—"
            style = "dim"
        else:
            pi_s = f"${pi:.2f}" if pi is not None else "—"
            po_s = f"${po:.2f}" if po is not None else "—"
            price_part = f"{pi_s}/{po_s}"
            style = "green" if i == 0 else None
        lines.append(f"{name:<13.13} ", style="cyan")
        lines.append(price_part, style=style)
    if len(pp) > limit:
        lines.append(f"\n+{len(pp) - limit} more", style="dim")
    return lines


def _speed_text(tps: float | None) -> Text:
    if tps is None:
        return Text("—", style="dim")
    if tps >= 80:
        style = "green"
    elif tps >= 30:
        style = "yellow"
    else:
        style = "red"
    return Text(f"{tps:5.0f}", style=style)


def _ttft_text(ms: float | None) -> Text:
    if ms is None:
        return Text("—", style="dim")
    s = ms / 1000.0
    if s < 1.0:
        style = "green"
    elif s < 3.0:
        style = "yellow"
    else:
        style = "red"
    return Text(f"{s:4.1f}s", style=style)


def _render_table(recs: list[Recommendation], c: Constraints) -> None:
    bits = [f"task=[cyan]{c.task}[/cyan]", f"mode=[cyan]{c.mode}[/cyan]"]
    if c.mode == "cloud":
        if c.max_price is not None:
            bits.append(f"max=[cyan]${c.max_price}/M[/cyan]")
        elif c.budget:
            bits.append(f"budget=[cyan]{c.budget}[/cyan]")
    else:
        bits.append(f"quant=[cyan]{c.quant}[/cyan]")
        if c.vram_gb:
            bits.append(f"vram=[cyan]{c.vram_gb}gb[/cyan]")
    title = "llm-rank  " + "  ".join(bits)

    table = Table(
        title=title,
        title_style="bold",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
        pad_edge=False,
        expand=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", overflow="fold", min_width=14, max_width=20)
    table.add_column("Quality", justify="left", width=14)
    table.add_column("Score", justify="right", width=7)
    if c.mode == "cloud":
        table.add_column("t/s", justify="right", width=5)
        table.add_column("TTFT", justify="right", width=5)
        table.add_column("Providers ($in/$out per M)", overflow="fold", min_width=24)
    else:
        table.add_column("VRAM", justify="right", width=8)
        table.add_column("Ctx", justify="right", style="dim", width=5)

    for r in recs:
        cost_or_vram: Text
        if c.mode == "cloud":
            cost_or_vram = _price_text(r.price_avg)
        else:
            cost_or_vram = (
                Text(f"{r.fits_vram_gb:4.1f} GB", style="green")
                if r.fits_vram_gb is not None
                else Text("—", style="dim")
            )

        ctx = "—"
        m = re.search(r"(\d+)k ctx", r.reason)
        if m:
            ctx = f"{m.group(1)}k"

        rank_text = Text(str(r.rank), style="bold yellow" if r.rank == 1 else "dim")

        row = [
            rank_text,
            _trunc(r.display_name, 20),
            _quality_bar(r.quality),
            _score_text(r.score),
        ]
        if c.mode == "cloud":
            row.append(_speed_text(r.throughput_tps))
            row.append(_ttft_text(r.latency_ms))
            row.append(_provider_pricing_text(r.provider_pricing))
        else:
            row.append(cost_or_vram)
            row.append(ctx)
        table.add_row(*row, end_section=(c.mode == "cloud"))
    console.print(table)

    # Ollama hints below the table for local mode (avoids row wrapping noise).
    if c.mode == "local":
        ollama_lines = [r for r in recs if r.ollama_cmd]
        if ollama_lines:
            console.print()
            for r in ollama_lines:
                console.print(f"  [dim]#{r.rank}[/dim] [cyan]{r.ollama_cmd}[/cyan]")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"llm-rank {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command("recommend")
def recommend_cmd(
    task: str = typer.Option("general", "--task", help="coding | general | math | reasoning"),
    mode: str = typer.Option("cloud", "--mode", help="cloud | local"),
    budget: Optional[str] = typer.Option(None, "--budget", help="low | med | high (cloud only)"),
    max_price: Optional[float] = typer.Option(None, "--max-price", help="$/M tokens cap (cloud)"),
    vram: Optional[str] = typer.Option(None, "--vram", help="e.g. 12gb (local; auto-detect if omitted)"),
    quant: str = typer.Option("q5_k_m", "--quant", help="fp16 | q8 | q5_k_m | q4_k_m"),
    top: int = typer.Option(3, "--top", help="Number of recommendations."),
    json_out: bool = typer.Option(False, "--json", help="Shortcut for --format json."),
    fmt: str = typer.Option("table", "--format", help=f"Output: {' | '.join(VALID_FORMATS)}"),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass cache and re-fetch sources."),
    rated_only: bool = typer.Option(False, "--rated-only", help="Hide models without HF leaderboard quality."),
) -> None:
    """Recommend top models for the given task and constraints."""
    if task not in ("coding", "general", "math", "reasoning"):
        raise typer.BadParameter(f"Unknown task: {task}")
    if mode not in ("cloud", "local"):
        raise typer.BadParameter(f"Unknown mode: {mode}")
    if quant not in ("fp16", "q8", "q5_k_m", "q4_k_m"):
        raise typer.BadParameter(f"Unknown quant: {quant}")
    if budget is not None and budget not in ("low", "med", "high"):
        raise typer.BadParameter(f"Unknown budget: {budget}")

    c = Constraints(
        task=task,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        budget=budget,  # type: ignore[arg-type]
        max_price=max_price,
        vram_gb=_parse_size_gb(vram),
        quant=quant,  # type: ignore[arg-type]
        top=top,
    )

    if json_out:
        fmt = "json"
    if fmt not in VALID_FORMATS:
        raise typer.BadParameter(f"Unknown format: {fmt}")

    recs = recommend.recommend(c, refresh=refresh, rated_only=rated_only)

    if fmt == "json":
        _emit_json(recs, c)
        return
    if fmt == "csv":
        _emit_csv(recs)
        return

    if not recs:
        console.print("[yellow]No models matched the given constraints.[/yellow]")
        raise typer.Exit(code=1)
    _render_table(recs, c)


@app.command("list")
def list_cmd(
    task: str = typer.Option("general", "--task"),
    mode: str = typer.Option("cloud", "--mode"),
    top: int = typer.Option(50, "--top", "-n", help="Max rows to show (use 0 / --all for unlimited)."),
    all_rows: bool = typer.Option(False, "--all", help="Show every match, ignore --top."),
    budget: Optional[str] = typer.Option(None, "--budget", help="low | med | high (cloud only)"),
    max_price: Optional[float] = typer.Option(None, "--max-price", help="$/M tokens cap (cloud)"),
    min_quality: Optional[float] = typer.Option(None, "--min-quality", help="0..100 quality floor"),
    name_filter: Optional[str] = typer.Option(None, "--filter", "-f", help="Substring match on model id or display name (e.g. 'openai', 'anth')."),
    fmt: str = typer.Option("table", "--format", help=f"Output: {' | '.join(VALID_FORMATS)}"),
    refresh: bool = typer.Option(False, "--refresh"),
    rated_only: bool = typer.Option(False, "--rated-only", help="Hide unrated models."),
) -> None:
    """List top models matching simple filters."""
    if fmt not in VALID_FORMATS:
        raise typer.BadParameter(f"Unknown format: {fmt}")
    if task not in ("coding", "general", "math", "reasoning"):
        raise typer.BadParameter(f"Unknown task: {task}")
    if mode not in ("cloud", "local"):
        raise typer.BadParameter(f"Unknown mode: {mode}")
    if budget is not None and budget not in ("low", "med", "high"):
        raise typer.BadParameter(f"Unknown budget: {budget}")
    requested_top = 10_000 if (all_rows or top == 0) else top
    # Post-filters (name/min_quality) need full universe to pick from.
    needs_full_scan = bool(name_filter) or min_quality is not None
    fetch_top = 10_000 if needs_full_scan else requested_top
    c = Constraints(
        task=task,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        budget=budget,  # type: ignore[arg-type]
        max_price=max_price,
        top=fetch_top,
    )
    recs = recommend.recommend(c, refresh=refresh, rated_only=rated_only)
    if min_quality is not None:
        recs = [r for r in recs if r.quality is not None and r.quality >= min_quality]
    if name_filter:
        needle = name_filter.lower()
        recs = [r for r in recs if needle in r.model_id.lower() or needle in r.display_name.lower()]
    # Truncate after post-filters so --filter + --top compose correctly.
    recs = recs[:requested_top]
    for i, r in enumerate(recs, 1):
        r.rank = i
    if fmt == "json":
        _emit_json(recs, c)
        return
    if fmt == "csv":
        _emit_csv(recs)
        return
    if not recs:
        console.print("[yellow]No models.[/yellow]")
        raise typer.Exit(code=1)
    _render_table(recs, c)


@app.command()
def info(model_id: str) -> None:
    """Show what we know about one model id."""
    hf_rows = recommend.load_hf()
    or_rows = recommend.load_openrouter()
    entries = recommend.merge(hf_rows, or_rows)
    needle = model_id.lower()
    for e in entries:
        if needle == e.model_id.lower() or needle in e.model_id.lower():
            console.print_json(json.dumps({
                "model_id": e.model_id,
                "display_name": e.display_name,
                "params_b": e.params_b,
                "quality": e.quality,
                "price_in_per_m": e.price_in,
                "price_out_per_m": e.price_out,
                "context_len": e.context_len,
                "available_cloud": e.available_cloud,
                "available_local": e.available_local,
            }))
            return
    console.print(f"[red]Not found:[/red] {model_id}")
    raise typer.Exit(code=1)


@app.command("hardware")
def hw() -> None:
    """Show detected RAM/VRAM."""
    info = hardware.detect()
    console.print(f"RAM:  {info.ram_gb:.1f} GB")
    if info.vram_gb is not None:
        console.print(f"VRAM: {info.vram_gb:.1f} GB  ({info.gpu_name or 'unknown GPU'})")
    else:
        console.print("VRAM: not detected (no CUDA torch / nvidia-smi)")


@app.command()
def update() -> None:
    """Force-refresh all source caches."""
    cache.clear()
    console.print("[dim]Fetching HuggingFace leaderboard...[/dim]")
    recommend.load_hf(refresh=True)
    console.print("[dim]Fetching LMArena Elo leaderboard...[/dim]")
    recommend.load_arena(refresh=True)
    console.print("[dim]Fetching OpenRouter frontend (perf data, ~750 models)...[/dim]")
    recommend.load_openrouter_frontend(refresh=True)
    console.print("[dim]Fetching OpenRouter public API (fallback)...[/dim]")
    recommend.load_openrouter(refresh=True)
    console.print("[green]Caches refreshed.[/green]")


if __name__ == "__main__":
    app()
