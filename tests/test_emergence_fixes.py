"""Regression tests for the adversarial-audit fixes on the emergence discovery pipeline.

Each test encodes a CONFIRMED audit finding's reproduction and asserts it is now fixed:
  P0  reason_cluster transient-error -> abstain (not cached, not 'junk'); hardened JSON parse.
  P1  load_full corpus cache invalidates when a new month-parquet appears.
  P2  granularity_gates verb guard drops finite verbs, keeps noun heads.
  P3  cooccur_cluster average-linkage does NOT chain unrelated themes through a bridge.
  P4  share_growth reads ~1x for a flat-share theme under corpus inflation.
"""
import pandas as pd
import pytest


# ---- P2: verb guard -------------------------------------------------------
def test_verb_guard_drops_verbs_keeps_nouns():
    from marketbrain.markets.granularity_gates import _looks_like_verb
    for v in ["destabilizes", "mediates", "emits", "outperforms", "leverages", "optimizes", "normalizes"]:
        assert _looks_like_verb(v), v
    for noun in ["rewards", "embeddings", "distillation", "agents", "size", "series", "analysis"]:
        assert not _looks_like_verb(noun), noun


# ---- P2b: tokenization-fragment gate --------------------------------------
def test_fragment_gate_drops_artifacts_keeps_real_names():
    from marketbrain.markets.granularity_gates import _is_fragment
    for junk in ["b--", "b-a", "gpt-oss-", "qwen b-a", "a-b-c"]:
        assert _is_fragment(junk), junk
    for real in ["k-means", "t-sne", "q-learning", "r-cnn", "u-net", "gpt-4", "on-policy distillation", "tool-call"]:
        assert not _is_fragment(real), real


# ---- P3: average-linkage anti-chaining ------------------------------------
def test_clustering_does_not_chain_through_bridge():
    from marketbrain.markets.cooccur_cluster import cooccurrence, cluster
    papers = ([frozenset({"a", "b"})] * 24 + [frozenset({"c", "b"})] * 24
              + [frozenset({"a"})] * 6 + [frozenset({"c"})] * 6)            # df: a=30, c=30, b=48; J(a,c)=0
    cooc, df = cooccurrence(papers)
    groups = [set(g) for g in cluster(["a", "b", "c"], cooc, df)]
    a_grp = next(g for g in groups if "a" in g)
    assert "c" not in a_grp, f"bridge 'b' chained 'a' and 'c': {groups}"


# ---- P4: share-based growth -----------------------------------------------
def test_share_growth_flat_theme_reads_one():
    from marketbrain.markets.granularity_gates import share_growth
    flat = share_growth(140, 100, 14000, 10000)        # 1% share in both windows, corpus grew 1.4x
    assert 0.9 < flat < 1.1, flat
    accel = share_growth(280, 50, 14000, 10000)        # share doubled+
    assert accel > 2.0, accel


# ---- P0: hardened parser reuse --------------------------------------------
@pytest.mark.parametrize("txt", [
    'Result:\n```json\n{"is_market": true, "coherence": 0.9}\n```\nLet me know if you need anything {else}.',
    '```json\n{"is_market": false}\n```\nWait, reconsider:\n```json\n{"is_market": true, "coherence": 0.8}\n```',
    'prose with a stray {a, b} brace, then {"is_market": true, "coherence": 0.7}',
])
def test_parser_survives_fences_and_prose(txt):
    from marketbrain.labels.gemini_grounding import _extract_json
    assert _extract_json(txt).get("is_market") is True


# ---- P0: transient error -> abstain, not cached, not junk ------------------
class _Flaky:
    class models:
        @staticmethod
        def generate_content(**_):
            raise RuntimeError("network blip")


class _Good:
    class models:
        @staticmethod
        def generate_content(**_):
            class R:
                text = '{"is_market": true, "market_name": "On-Policy Distillation", "member_phrases": ["opd"], "coherence": 0.9}'
            return R()


def test_transient_error_abstains_and_is_not_cached(monkeypatch, tmp_path):
    import marketbrain.markets.reason_cluster as rc
    monkeypatch.setattr(rc, "RETRIES", 1)            # no backoff sleep
    monkeypatch.setattr(rc, "CACHE", tmp_path / "cache.jsonl")
    res = rc.name_clusters([(["on-policy distillation", "opd"], ["t"])], client=_Flaky(), use_cache=True)
    assert res[0]["status"] == "error" and res[0]["is_market"] is False
    assert not (tmp_path / "cache.jsonl").exists(), "abstention must NOT be cached"


def test_good_response_parses_and_enforces_subset(monkeypatch, tmp_path):
    import marketbrain.markets.reason_cluster as rc
    monkeypatch.setattr(rc, "CACHE", tmp_path / "cache.jsonl")
    res = rc.name_clusters([(["opd", "on-policy distillation"], ["t"])], client=_Good(), use_cache=True)
    assert res[0]["status"] == "ok" and res[0]["is_market"] is True
    assert res[0]["member_phrases"] == ["opd"]       # subset of input enforced
    assert (tmp_path / "cache.jsonl").exists(), "definitive verdict must be cached"


# ---- P1: corpus cache invalidates on a new month --------------------------
def test_corpus_cache_invalidates_on_new_month(monkeypatch, tmp_path):
    import marketbrain.corpus.load_full as lf
    raw, der = tmp_path / "raw", tmp_path / "der"
    (raw / "cs.AI").mkdir(parents=True)
    der.mkdir()
    monkeypatch.setattr(lf, "RAW", raw)
    monkeypatch.setattr(lf, "DERIVED", der)
    cols = ["record_id", "month", "published", "title", "abstract", "authors",
            "categories", "primary_category", "comment", "journal_ref", "doi"]

    def mk(rid, mon, path):
        row = {c: "" for c in cols}
        row["record_id"], row["month"] = rid, mon
        pd.DataFrame([row]).to_parquet(path)

    mk("a", "2020-01", raw / "cs.AI" / "2020-01.parquet")
    assert len(lf.load_corpus()) == 1                # builds cache
    mk("b", "2020-02", raw / "cs.AI" / "2020-02.parquet")
    assert len(lf.load_corpus()) == 2                # new month -> cache rebuilt, not stale
