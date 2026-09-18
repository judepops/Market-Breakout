import json

from experiments.llm_concepts import (
    build_request,
    estimate_cost,
    normalize_concepts,
    parse_result_line,
)


def test_build_request_shape_and_key():
    req = build_request("arxiv:1", "A Title", "An abstract.")
    assert req["key"] == "arxiv:1"
    body = req["request"]
    assert "A Title" in body["contents"][0]["parts"][0]["text"]
    gc = body["generation_config"]
    assert gc["response_mime_type"] == "application/json"
    assert gc["response_schema"]["required"] == ["concepts"]


def test_build_request_truncates_huge_abstract():
    req = build_request("id", "t", "x" * 100_000)
    assert len(req["request"]["contents"][0]["parts"][0]["text"]) < 10_000


def test_normalize_concepts_cleans_and_dedupes():
    raw = {"concepts": ["  Chain-of-Thought Prompting ", "KV-Cache", "kv-cache",
                        "a very long concept name with too many words", 42, "x"]}
    out = normalize_concepts(raw)
    assert out == ["chain-of-thought prompting", "kv-cache"]


def test_normalize_concepts_handles_garbage():
    assert normalize_concepts(None) == []
    assert normalize_concepts({"nope": []}) == []


def test_parse_result_line_success_and_error():
    ok = json.dumps({
        "key": "arxiv:9",
        "response": {"candidates": [{"content": {"parts": [
            {"text": json.dumps({"concepts": ["speculative decoding"]})}]}}]},
    })
    assert parse_result_line(ok) == ("arxiv:9", ["speculative decoding"])

    err = json.dumps({"key": "arxiv:8", "error": {"message": "boom"}})
    assert parse_result_line(err) is None

    malformed_payload = json.dumps({
        "key": "arxiv:7",
        "response": {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]},
    })
    assert parse_result_line(malformed_payload) == ("arxiv:7", [])


def test_estimate_cost_scales():
    e = estimate_cost(184_335)
    assert e["papers"] == 184_335
    assert 80 < e["est_cost_usd_batch"] < 130  # ~$101 at current batch pricing
