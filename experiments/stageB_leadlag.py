"""Stage B test #3 (pre-registered): TEMPORAL LEAD-LAG per market -- does research rise BEFORE capital?

    python -m experiments.stageB_leadlag

Distinct from the cross-sectional #1/#2: within each market, does the monthly RESEARCH series lead the monthly
CAPITAL series in time? Cross-correlate on GROWTH (differenced, stationary) series -- trending levels would
manufacture a spurious lead. Per market, lag* = argmax_l corr(dResearch[t], dCapital[t+l]) over l in [-12,12];
research leads iff lag* > 0. Aggregate median lag* + a circular-shift shuffle null (breaks temporal alignment
while preserving each series' autocorrelation). Prior s-curve lead-lag nulled at p~=0.27; one honest retest.
"""
from __future__ import annotations
import os
import pathlib

import numpy as np
import pandas as pd

SB = pathlib.Path("artifacts/stageB")
TAU = float(os.environ.get("BRIDGE_TAU", "0.63"))
LAGMAX = 12
MIN_CAP_MONTHS = 8          # need enough nonzero capital months for a meaningful cross-correlation


def _months(m):
    return int(m[:4]) * 12 + (int(m[5:7]) - 1)


def _embed(texts):
    from marketbrain.labels.gemini_grounding import make_client
    client = make_client()
    out = []
    for i in range(0, len(texts), 100):
        r = client.models.embed_content(model="gemini-embedding-001", contents=[str(t)[:2000] for t in texts[i:i + 100]])
        out.extend(np.asarray(e.values, dtype=np.float32) for e in r.embeddings)
    X = np.vstack(out)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _best_lag(dr, dc):
    """lag* in [-LAGMAX,LAGMAX] maximising corr(dr[t], dc[t+lag]); +ve => research leads capital."""
    best_l, best_c = 0, -2.0
    for l in range(-LAGMAX, LAGMAX + 1):
        if l >= 0:
            a, b = dr[:len(dr) - l], dc[l:]
        else:
            a, b = dr[-l:], dc[:len(dc) + l]
        if len(a) < 6 or np.std(a) == 0 or np.std(b) == 0:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if not np.isnan(c) and c > best_c:
            best_c, best_l = c, l
    return best_l, best_c


def main():
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.markets.granularity_gates import _noun_headed, _is_fragment, GENERIC

    themes_file = os.environ.get("THEMES_FILE", "clustered_markets.parquet")
    th = pd.read_parquet(SB / themes_file)
    if "members" in th.columns:                                  # clustered layer -> member phrases
        th["members_list"] = th["members"].fillna("").apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    else:                                                        # single-phrase layer -> the phrase is its own market
        th["members_list"] = th["theme_id"].apply(lambda t: [t])

    # research monthly series per market (reload corpus; recompute permissive te; union member phrases)
    corpus = load_corpus("cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML".split(","))
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    print(f"[leadlag] {len(corpus):,} papers; building research series for {len(th)} markets...", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    allmem = set().union(*th["members_list"])
    te = te[te["term"].isin(allmem)]
    rec_month = te.drop_duplicates("record_id").set_index("record_id")["month"]
    term_recs = te.groupby("term")["record_id"].apply(set).to_dict()

    def research_series(members):
        recs = set().union(*[term_recs.get(t, set()) for t in members]) if members else set()
        mo = pd.Series([rec_month.get(r) for r in recs]).value_counts()
        return pd.Series(0, index=grid, dtype=float).add(mo.reindex(grid).fillna(0), fill_value=0)

    # capital monthly series per market via cached bridge
    comp = pd.read_parquet("data/raw/labels/company_descriptions.parquet")
    comp = comp[comp["bridge_text"].str.len() > 0].reset_index(drop=True)
    tvec = _embed(list(th["theme_text"]))
    cvec = np.load(SB / "company_emb.npy"); ids = list(np.load(SB / "company_ids.npy", allow_pickle=True))
    assert ids == list(comp["permalink"]), "company embedding cache out of sync"
    sims = cvec @ tvec.T
    best = sims.argmax(1)
    comp = comp.assign(theme_id=th["theme_id"].to_numpy()[best], sim=sims[np.arange(len(comp)), best])
    matched = comp[comp["sim"] >= TAU][["permalink", "theme_id"]]
    rounds = pd.read_parquet("data/raw/labels/company_rounds.parquet").dropna(subset=["announced_on"])
    rounds["month"] = rounds["announced_on"].astype(str).str[:7]
    mr = matched.merge(rounds, on="permalink", how="inner")

    def capital_series(mid):
        s = mr[mr["theme_id"] == mid].groupby("month")["raised_usd"].sum()
        return pd.Series(0, index=grid, dtype=float).add(s.reindex(grid).fillna(0), fill_value=0)

    lags, caps = [], {}
    for _, row in th.iterrows():
        mid = row["theme_id"]
        C = capital_series(mid)
        if (C > 0).sum() < MIN_CAP_MONTHS:
            continue
        R = research_series(row["members_list"])
        # trim to the market's active span (first research month .. last capital month)
        active = [i for i, m in enumerate(grid) if R.iloc[i] > 0 or C.iloc[i] > 0]
        if len(active) < 24:
            continue
        lo, hi = active[0], active[-1] + 1
        dr = np.diff(np.log1p(R.to_numpy()[lo:hi]))
        dc = np.diff(np.log1p(C.to_numpy()[lo:hi]))
        l, c = _best_lag(dr, dc)
        lags.append(l); caps[mid] = (dr, dc)
    lags = np.array(lags)
    if len(lags) < 8:
        print(f"[leadlag] only {len(lags)} markets with enough capital months -- underpowered"); return
    med = float(np.median(lags))
    frac_lead = float((lags > 0).mean())

    # circular-shift null: shift each market's capital-growth by a random offset, recompute median lag*
    rng = np.random.RandomState(0)
    null_meds = []
    keys = list(caps)
    for _ in range(3000):
        nl = []
        for mid in keys:
            dr, dc = caps[mid]
            sh = rng.randint(1, len(dc))
            l, _ = _best_lag(dr, np.roll(dc, sh))
            nl.append(l)
        null_meds.append(np.median(nl))
    null_meds = np.array(null_meds)
    p = (np.sum(null_meds >= med) + 1) / (len(null_meds) + 1)

    print(f"\n[leadlag] === TEMPORAL LEAD-LAG ({len(lags)} markets, lag>0 => research leads capital) ===")
    print(f"   median lag* = {med:+.1f} months   frac markets research-leads = {frac_lead:.0%}")
    print(f"   circular-shift null: median lag* null mean={null_meds.mean():+.2f}  ->  p(median>=obs) = {p:.3f}"
          f"{'  <-- research LEADS capital' if (p < 0.05 and med > 0) else '  (null: no lead)'}", flush=True)
    print(f"   lag* distribution: {dict(zip(*np.unique(lags, return_counts=True)))}", flush=True)


if __name__ == "__main__":
    main()
