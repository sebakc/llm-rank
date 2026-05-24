from llm_rank import recommend
from llm_rank.models import Constraints


def test_merge_attaches_hf_quality_to_openrouter(hf_rows, or_rows):
    entries = recommend.merge(hf_rows, or_rows)
    by_id = {e.model_id: e for e in entries}
    # llama-3-8b OpenRouter id should pick up HF quality via slug match.
    llama = by_id.get("meta-llama/llama-3-8b-instruct")
    assert llama is not None
    assert llama.available_cloud
    assert llama.quality.get("general") == 68.4


def _patch_sources(monkeypatch, hf_rows, or_rows):
    monkeypatch.setattr(recommend, "load_hf", lambda refresh=False: hf_rows)
    monkeypatch.setattr(recommend, "load_arena", lambda refresh=False: [])
    monkeypatch.setattr(recommend, "load_openrouter_frontend", lambda refresh=False: or_rows)
    monkeypatch.setattr(recommend, "load_openrouter", lambda refresh=False: or_rows)


def test_recommend_cloud_budget_low_filters(monkeypatch, hf_rows, or_rows):
    _patch_sources(monkeypatch, hf_rows, or_rows)
    c = Constraints(task="coding", mode="cloud", budget="low", top=5)
    recs = recommend.recommend(c, rated_only=True)
    assert recs, "expected at least one cheap coding recommendation"
    # GPT-4o ($10/M avg) must be filtered by low budget (cap $1.0/M).
    assert all("gpt-4o" not in r.model_id.lower() for r in recs)


def test_recommend_local_vram_filter(monkeypatch, hf_rows, or_rows):
    _patch_sources(monkeypatch, hf_rows, or_rows)
    c = Constraints(task="general", mode="local", vram_gb=8.0, quant="q5_k_m", top=5)
    recs = recommend.recommend(c)
    # 16B model at q5 needs ~16*0.7 + 1.5 = 12.7 GB; must be excluded at 8 GB.
    assert all("deepseek-coder-v2" not in r.model_id for r in recs)
    # 7B/8B should fit (~7.4 GB or ~7.1 GB <= 8).
    ids = " ".join(r.model_id.lower() for r in recs)
    assert "llama" in ids or "mistral" in ids


def test_recommend_top_n(monkeypatch, hf_rows, or_rows):
    _patch_sources(monkeypatch, hf_rows, or_rows)
    c = Constraints(task="general", mode="cloud", top=2)
    recs = recommend.recommend(c)
    assert len(recs) <= 2
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))
