"""Offline smoke test: the house-style plotter runs and writes a figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — no display in tests

from experiments.plot_signal import main as plot_main
from marketbrain.corpus import CorpusStore
from marketbrain.schema import observations_to_frame
from marketbrain.signals.arxiv import count_to_observation


def test_plot_writes_png_and_pdf(tmp_path):
    store = CorpusStore(tmp_path)
    obs = [count_to_observation(m, c, "http://x", market="llm-dev-tools")
           for m, c in [("2024-01", 10), ("2024-02", 14), ("2024-03", 21)]]
    store.append_observations(observations_to_frame(obs))

    out = tmp_path / "figs"
    rc = plot_main(["arxiv", "--root", str(tmp_path), "--market", "llm-dev-tools", "--out", str(out)])
    assert rc == 0
    assert (out / "arxiv_llm-dev-tools_trajectory.png").exists()
    assert (out / "arxiv_llm-dev-tools_trajectory.pdf").exists()
