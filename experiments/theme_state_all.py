"""Current research state for EVERY theme + predictions for ALL recent themes.

Two things the labeled radar cannot do, both computable now:
 (1) TRAJECTORY (accelerating / sustained / cooling / faded) for every labeled theme,
     including pre-2023 ones, from live corpus activity to 2026-06. Uses SHARE of the
     corpus (not raw counts) so it is not confounded by overall arXiv growth.
 (2) PREDICTIONS for all recent (2023-2026) emergent themes the labeled pipeline
     censors: scored with the frozen live model, no outcome label.

Headline 6-cat corpus to 2026-06 (matches the validated model). Writes
artifacts/theme_state/{trajectory.json, predictions.json, validation.json}.

    python -m experiments.theme_state_all
"""
from __future__ import annotations
import json, pathlib, warnings
warnings.filterwarnings("ignore")

MB = pathlib.Path(".")
APP = pathlib.Path(__import__("os").environ.get("APP_BACKEND_DIR", "../app/backend"))
OUT = MB / "artifacts/theme_state"
CATS = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]

DIMS = [
    ("venue_appearance", "Venue penetration", "Did early papers reach strong venues (NeurIPS etc.)? The top driver."),
    ("persistence_frac", "Persistence", "How much of its early life the idea stayed active, window-normalised."),
    ("escore_slope", "Emergence slope", "How steeply activity was rising at the decision point."),
    ("author_breadth", "Author breadth", "How many distinct authors adopted it early, broad uptake not one lab."),
    ("accel_2nd_deriv", "Acceleration", "Is the growth itself speeding up, rather than plateauing?"),
    ("author_inflow", "New-author inflow", "Share of recent authors new to the idea."),
    ("volume_share", "Volume share", "The idea's share of the whole field's recent output."),
    ("growth_ratio", "Growth ratio", "Raw growth of activity. High growth alone is a mild fad signature."),
    ("escore_norm", "Emergence level", "Overall emergence magnitude, size-normalised."),
    ("cross_domain_spread", "Cross-domain spread", "How many fields it touches. Near-random in the model."),
    ("cross_listing_rate", "Cross-listing", "How often its papers span multiple areas. Essentially no effect."),
]


def main():
    import numpy as np, pandas as pd, joblib, bisect
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.eval.breakout_signature import theme_features

    cfg = EmergenceLabelConfig()
    corpus = load_corpus(CATS)
    denom = cs_denominator(corpus); grid = list(denom.index); corpus_end = grid[-1]; nP = len(corpus)
    latest = corpus_end
    print(f"[state] {nP:,} papers, grid {grid[0]}..{corpus_end}", flush=True)
    total_by_month = denom.astype(float)

    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    dfc = te.groupby("term")["record_id"].nunique().to_dict(); ceil = DF_CEIL * nP
    gated = [t for t, c in dfc.items() if c <= ceil and not _is_fragment(t)
             and not (len(t.split()) == 1 and t in GENERIC)
             and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split())) and _pmi_ok(t, dfc, nP)]
    gated = set(_noun_headed(gated)); te = te[te["term"].isin(gated)]
    pivot = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)
    da = denom.reindex(grid).to_numpy(dtype=float)
    share_pivot = pivot.divide(np.where(da > 0, da, 1.0), axis=1)   # theme share of the corpus per month

    # T0 + label for every gated term (corpus_end = 2026-06)
    lab = {}
    for term, nth in pivot.iterrows():
        share = pd.Series(np.divide(nth.to_numpy(), da, out=np.zeros(len(grid)), where=da > 0), index=grid)
        res = durable_breakout_label(share, nth.astype(int), cfg, corpus_end=corpus_end)
        lab[term] = (res.get("T0"), res["label"])
    print(f"[state] labelled {len(lab):,} gated terms", flush=True)

    gi = {m: i for i, m in enumerate(grid)}

    def trajectory(term, t0):
        if t0 not in gi:
            return None
        s = share_pivot.loc[term].iloc[gi[t0]:]
        if len(s) < 3:
            return None
        sm = s.rolling(3, center=True, min_periods=1).mean()
        peak_val = float(sm.max()); peak_month = str(sm.idxmax())
        recent = float(sm.iloc[-7:-1].mean()) if len(sm) >= 7 else float(sm.iloc[-3:].mean())
        retention = recent / peak_val if peak_val > 0 else 0.0
        msp = gi[latest] - gi[peak_month]
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
        return {"state": state, "retention": round(retention, 2), "peak_month": peak_month,
                "months_since_peak": int(msp), "recent_slope": round(slope, 3)}

    # ---------- Part 1: trajectory for the labelled radar themes ----------
    labeled = pd.read_parquet(MB / "artifacts/phase1_aiml/scores.parquet")
    traj_out, val_rows = {}, []
    for r in labeled.itertuples(index=False):
        t0 = lab.get(r.theme_id, (r.T0, None))[0] or r.T0
        tj = trajectory(r.theme_id, t0)
        if tj:
            traj_out[r.theme_id] = tj
            val_rows.append((r.label, tj["state"]))
    # validation: do durable read sustained/accelerating and fads read cooling/faded?
    vd = pd.DataFrame(val_rows, columns=["label", "state"])
    alive = {"accelerating", "sustained"}; dying = {"cooling", "faded"}
    dur = vd[vd.label == "positive"]; fad = vd[vd.label == "hyped_died"]
    validation = {
        "n": len(vd),
        "durable_alive_pct": round(100 * dur.state.isin(alive).mean(), 1),
        "fad_dying_pct": round(100 * fad.state.isin(dying).mean(), 1),
        "durable_state_mix": dur.state.value_counts().to_dict(),
        "fad_state_mix": fad.state.value_counts().to_dict(),
    }
    print(f"[state] Part 1: trajectory for {len(traj_out)} labelled themes", flush=True)
    print(f"        VALIDATION durable->alive {validation['durable_alive_pct']}%  |  fad->dying {validation['fad_dying_pct']}%", flush=True)

    # ---------- Part 2: predictions for all recent emergent themes ----------
    coefs = json.loads((APP / "mb_coefs.json").read_text())
    dist = json.loads((APP / "feat_percentiles.json").read_text())
    bundle = joblib.load(MB / "artifacts/live_scorer/model.joblib")
    clf, FEATS = bundle["model"], bundle["features"]
    tr = pd.read_parquet(MB / "artifacts/live_scorer/train_feats_frac.parquet")
    tr = tr[tr["label"].isin(("positive", "hyped_died"))]
    allp = clf.predict_proba(tr[FEATS].fillna(0).to_numpy(float))[:, 1]
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    crecs = corpus.set_index("record_id")

    def pct_of(feat, v):
        vs = dist.get(feat); return round(bisect.bisect_right(vs, v) / len(vs) * 100, 0) if vs else 0.0

    def contrib(feat, v):
        c = coefs["features"].get(feat); return round(c["coef"] * (v - c["mean"]) / (c["std"] + 1e-9), 3) if c else 0.0

    def fv(f):
        f = dict(f); f["persistence_frac"] = f["persistence_months"] / max(f.get("window_len", 1), 1); return f

    recent = [(t, t0) for t, (t0, _lb) in lab.items()
              if t0 and "2023-03" <= t0 <= "2025-09" and t0 in gi and gi[t0] + 9 < len(grid)]
    print(f"[state] Part 2: scoring {len(recent)} recent (2023-2026) emergent themes...", flush=True)
    preds = []
    for term, t0 in recent:
        recs = crecs.loc[list(term_to_records[term])].reset_index()
        t_dec = grid[gi[t0] + 9]
        f = fv(theme_features(recs, t_dec, total_by_month))
        if f.get("window_len", 0) < 3:
            continue
        v = np.array([[f[k] for k in FEATS]], float)
        p = float(clf.predict_proba(v)[0, 1]); pctile = float((allp <= p).mean() * 100)
        rating = "L5" if pctile >= 90 else "L4" if pctile >= 70 else "L3" if pctile >= 45 else "L2" if pctile >= 25 else "L1"
        dims = [{"label": lb, "score": pct_of(k, f[k]), "contrib": contrib(k, f[k]), "desc": de}
                for k, lb, de in DIMS if k in f]
        preds.append({
            "theme_id": term, "t0": t0, "t_dec": t_dec, "rating": rating,
            "durability": round(pctile, 1), "sig_score": round(p, 3), "dims": dims,
            "trajectory": trajectory(term, t0),
            "n_papers_asof_decision": int(recs[recs["month"] <= t_dec]["record_id"].nunique()),
            "n_papers_now": int(recs["record_id"].nunique()),
        })
    preds.sort(key=lambda d: -d["durability"])
    print(f"[state] Part 2: {len(preds)} recent predictions "
          f"({sum(p['rating'] in ('L4','L5') for p in preds)} at L4/L5)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trajectory.json").write_text(json.dumps(traj_out, indent=1))
    (OUT / "predictions.json").write_text(json.dumps(preds, indent=1))
    (OUT / "validation.json").write_text(json.dumps(validation, indent=1))
    print(f"[state] wrote {OUT}/ (trajectory {len(traj_out)}, predictions {len(preds)})", flush=True)


if __name__ == "__main__":
    main()
