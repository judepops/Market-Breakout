import numpy as np

from experiments.concept_themes import merge_concepts, norm_plural


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_norm_plural():
    assert norm_plural("state space models") == norm_plural("state space model")
    assert norm_plural("vlms") == norm_plural("vlm")
    assert norm_plural("cot") == "cot"  # short words untouched
    # known limitation: 'bias' also strips to 'bia'; harmless since 'bia' never
    # occurs as a real concept, and merges pick the higher-df spelling anyway


def test_merge_plural_variants_without_embeddings():
    concepts = ["vlm", "vlms", "graph neural networks"]
    df = {"vlm": 5, "vlms": 20, "graph neural networks": 30}
    m = merge_concepts(concepts, df, embeddings=None)
    assert m["vlm"] == "vlms"  # higher-df spelling wins
    assert m["vlms"] == "vlms"
    assert m["graph neural networks"] == "graph neural networks"


def test_merge_embedding_near_duplicates():
    concepts = ["chain-of-thought prompting", "cot prompting", "speculative decoding"]
    df = {"chain-of-thought prompting": 50, "cot prompting": 10, "speculative decoding": 40}
    e_cot = _unit([1.0, 0.02, 0.0])
    e_cot2 = _unit([1.0, 0.0, 0.02])       # cosine ~0.999 with e_cot
    e_spec = _unit([0.0, 1.0, 0.0])        # orthogonal
    m = merge_concepts(concepts, df, embeddings=np.stack([e_cot, e_cot2, e_spec]))
    assert m["cot prompting"] == "chain-of-thought prompting"
    assert m["speculative decoding"] == "speculative decoding"


def test_merge_is_transitive():
    concepts = ["foo model", "foo models", "foo model variant"]
    df = {"foo model": 3, "foo models": 9, "foo model variant": 1}
    # 'foo model' ~ 'foo models' via plural; 'foo model variant' ~ 'foo model'
    # via embedding -> all three collapse into one group
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])                  # plural match handles this pair
    e3 = _unit([1.0, 0.01])                 # near e1
    m = merge_concepts(concepts, df, embeddings=np.stack([e1, e2, e3]))
    assert len(set(m.values())) == 1
    assert set(m.values()) == {"foo models"}  # highest df is canonical
