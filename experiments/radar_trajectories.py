"""Compute a CURRENT trajectory (accelerating / sustained / cooling / faded) for every radar theme, using
corpus activity right up to the corpus end (~2026-06), the same metric the live panel uses. Doubles as a
validation: on historical themes we KNOW the durable/fad label, so the trajectory should agree.

    python -m experiments.radar_trajectories

Writes App/radar/theme_trajectories.json (theme_id -> {state, retention, peak_month, msp}) and prints the
label-vs-trajectory agreement.
"""
from __future__ import annotations
import json, pathlib
import numpy as np, pandas as pd

RADAR = pathlib.Path(__import__("os").environ.get("RADAR_DIR", "app/radar"))
CATS = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]


def traj(counts: pd.Series, latest_idx: int):
    sm = counts.rolling(3, center=True, min_periods=1).mean()
    if sm.max() <= 0:
        return {"state": "faded", "retention": 0.0, "peak_month": None, "msp": None}
    peak_val = float(sm.max()); peak_month = str(sm.idxmax())
    recent = float(sm.iloc[-7:-1].mean()) if len(sm) >= 7 else float(sm.iloc[-3:].mean())
    retention = recent / peak_val if peak_val > 0 else 0.0
    months = list(counts.index)
    msp = months.index(months[-1]) - months.index(peak_month)
    tail = sm.iloc[-12:].to_numpy()
    slope = float(np.polyfit(range(len(tail)), tail, 1)[0] / (tail.mean() + 1e-9)) if len(tail) >= 3 else 0.0
    if retention < 0.30:
        state = "faded"
    elif msp <= 6 and slope > 0.03:
        state = "accelerating"
    elif (msp >= 12 and slope < 0) or slope < -0.04:
        state = "cooling"
    else:
        state = "sustained"
    return {"state": state, "retention": round(retention, 2), "peak_month": peak_month, "msp": int(msp)}


def main():
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL

    data = json.loads(RADAR.joinpath("radar_data.json").read_text())
    themes = {t["id"]: t for t in data["themes"]}
    print(f"[traj] {len(themes)} radar themes to score", flush=True)

    corpus = load_corpus(CATS)
    denom = cs_denominator(corpus); grid = list(denom.index); corpus_end = grid[-1]; nP = len(corpus)
    print(f"[traj] corpus grid {grid[0]}..{corpus_end}", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    te = te[te["term"].isin(set(themes))]                                 # only the radar terms
    pivot = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)

    out = {}
    for tid, row in themes.items():
        if tid not in pivot.index:
            continue
        t0 = row.get("t0") or grid[0]
        series = pivot.loc[tid]
        series = series[[m for m in grid if m >= t0]]                     # from birth to corpus end
        if len(series) < 6:
            continue
        out[tid] = traj(series.astype(float), len(series))
    RADAR.joinpath("theme_trajectories.json").write_text(json.dumps(out))
    print(f"[traj] wrote {len(out)} trajectories", flush=True)

    # validation: does trajectory agree with the known durable/fad label?
    rows = [(themes[t]["class"], out[t]["state"]) for t in out if t in themes]
    df = pd.DataFrame(rows, columns=["label", "state"])
    print("\n[traj] label vs trajectory (row = known label, col = observed trajectory now):")
    print(pd.crosstab(df["label"], df["state"]))
    dur_good = df[(df.label == "positive") & (df.state.isin(["accelerating", "sustained"]))].shape[0] / max((df.label == "positive").sum(), 1)
    fad_bad = df[(df.label == "hyped_died") & (df.state.isin(["cooling", "faded"]))].shape[0] / max((df.label == "hyped_died").sum(), 1)
    print(f"\n[traj] durable themes reading sustained/accelerating: {dur_good:.0%}")
    print(f"[traj] fad themes reading cooling/faded:              {fad_bad:.0%}")


if __name__ == "__main__":
    main()
