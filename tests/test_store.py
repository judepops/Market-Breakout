"""Parquet store round-trips and deterministic cut hashing."""

from __future__ import annotations

from datetime import date

from marketbrain.corpus import CorpusStore, build_panel, cut_hash
from tests.conftest import make_obs, obs_frame


def test_observation_append_and_load_roundtrip(tmp_path):
    store = CorpusStore(tmp_path)
    store.append_observations(obs_frame([make_obs(value=10.0), make_obs(value=11.0)]))
    store.append_observations(obs_frame([make_obs(value=12.0)]))
    loaded = store.load_observations()
    assert len(loaded) == 3
    assert set(loaded["value"]) == {10.0, 11.0, 12.0}


def test_load_empty_store_returns_typed_empty(tmp_path):
    store = CorpusStore(tmp_path)
    df = store.load_observations()
    assert df.empty and "knowledge_date" in df.columns


def test_write_cut_records_manifest_and_roundtrips(tmp_path):
    store = CorpusStore(tmp_path)
    obs = obs_frame([make_obs(value=10.0, reference_date=date(2024, 1, 10),
                              knowledge_date=date(2024, 1, 31))])
    panel = build_panel(obs, "2024-01")
    h = store.write_cut("2024-01", panel)

    assert store.load_cut("2024-01")["value"].tolist() == [10.0]
    manifest = (tmp_path / "MANIFEST.tsv").read_text()
    assert "2024-01" in manifest and h in manifest


def test_cut_hash_is_order_independent(tmp_path):
    obs = obs_frame([
        make_obs(feature="a", value=1.0, reference_date=date(2024, 1, 1),
                 knowledge_date=date(2024, 1, 31)),
        make_obs(feature="b", value=2.0, reference_date=date(2024, 1, 1),
                 knowledge_date=date(2024, 1, 31)),
    ])
    panel = build_panel(obs, "2024-01")
    assert cut_hash(panel) == cut_hash(panel.iloc[::-1])
