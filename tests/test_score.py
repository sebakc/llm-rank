from llm_rank.models import Constraints, ModelEntry
from llm_rank.score import normalize_price, normalize_quality, rank, score_entry


def test_normalize_quality_clamps():
    assert normalize_quality(None) == 0.0
    assert normalize_quality(0) == 0.0
    assert normalize_quality(100) == 1.0
    assert normalize_quality(150) == 1.0
    assert normalize_quality(50) == 0.5


def test_normalize_price_log_scale_orders():
    # log-scaled: cheaper -> smaller value, capped at 1.0 when at max.
    cheap = normalize_price(0.1, 100.0)
    pricey = normalize_price(50.0, 100.0)
    maxed = normalize_price(100.0, 100.0)
    assert 0.0 < cheap < pricey < maxed == 1.0


def test_normalize_price_missing_is_neutral():
    assert normalize_price(None, 10.0) == 0.5
    assert normalize_price(5.0, None) == 0.5


def test_rank_orders_by_score_desc():
    entries = [
        ModelEntry(model_id="a", display_name="A", quality={"general": 80.0}, price_in=1.0, price_out=1.0, available_cloud=True),
        ModelEntry(model_id="b", display_name="B", quality={"general": 60.0}, price_in=0.1, price_out=0.1, available_cloud=True),
        ModelEntry(model_id="c", display_name="C", quality={"general": 90.0}, price_in=10.0, price_out=10.0, available_cloud=True),
    ]
    c = Constraints(task="general", mode="cloud")
    out = rank(entries, c)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
    # A balances quality + cost; C is highest quality but priciest, gets penalized
    # at the default cost weight. A wins; C drops to last.
    assert out[0][0].model_id == "a"
    assert out[-1][0].model_id == "c"


def test_score_entry_quality_fallback_to_general():
    e = ModelEntry(model_id="x", display_name="x", quality={"general": 70.0}, available_cloud=True)
    c = Constraints(task="coding", mode="cloud")
    s = score_entry(e, c, max_price=None, max_tps=None)
    # Should not be zero — quality fell back to general (70 -> 0.7).
    assert s > 0
