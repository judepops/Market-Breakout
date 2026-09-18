"""Score a BRAND-NEW research theme with the trained durable-breakout detector — the actual use case.

    python -m experiments.score_new_theme "abliteration" --out ../App/radar/new_theme_abliteration.json

Pulls the term's arXiv footprint, computes the 11 structural features STRICTLY as-of T0+9 with the *same*
`theme_features` code the trained model uses, then scores it against a model trained on the existing
phase1_aiml themes. Reports the as-of-9-month call (the honest decision point) and, for context, an
"as of today" re-score on the fuller window. Category denominator for post-corpus months is fetched live
from arXiv. This is a real, out-of-sample forward call — the theme emerged after the corpus ends (2023-02).
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd

MB = __import__("pathlib").Path(__file__).resolve().parents[1]
CATS = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]   # the phase1_aiml core
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}


DENOM_CACHE = MB / "artifacts/_arxiv_cat_denominator.json"


def _get(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if getattr(e, "code", None) == 429 and k < tries - 1:
                time.sleep(10 * (k + 1)); continue          # rate-limited: back off hard
            if k == tries - 1:
                raise
            time.sleep(4)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(4)


def fetch_papers(term):
    """All arXiv papers matching the term -> list of dicts (id, month, authors, categories, comment, journal_ref)."""
    out, start = [], 0
    while True:
        q = urllib.parse.quote(f'all:"{term}"' if " " in term else f"all:{term}")
        raw = _get(f"http://export.arxiv.org/api/query?search_query={q}&start={start}&max_results=100"
                   f"&sortBy=submittedDate&sortOrder=ascending")
        root = ET.fromstring(raw)
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        for e in entries:
            out.append({
                "record_id": e.find("a:id", NS).text,
                "month": e.find("a:published", NS).text[:7],
                "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                "categories": [c.get("term") for c in e.findall("a:category", NS)],
                "comment": (e.find("arxiv:comment", NS).text if e.find("arxiv:comment", NS) is not None else None),
                "journal_ref": (e.find("arxiv:journal_ref", NS).text if e.find("arxiv:journal_ref", NS) is not None else None),
            })
        start += 100
        if len(entries) < 100:
            break
        time.sleep(3)
    return out


def month_iter(m0, m1):
    y, mo = int(m0[:4]), int(m0[5:7]); out = []
    while f"{y:04d}-{mo:02d}" <= m1:
        out.append(f"{y:04d}-{mo:02d}"); mo += 1
        if mo > 12: mo = 1; y += 1
    return out


def cat_count(month):
    """Live arXiv count of core-category submissions in a month (the denominator)."""
    y, mo = int(month[:4]), int(month[5:7])
    nd = [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1]
    catq = "+OR+".join(f"cat:{c}" for c in CATS)
    q = f"({catq})+AND+submittedDate:[{y}{mo:02d}010000+TO+{y}{mo:02d}{nd}2359]"
    raw = _get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=1")
    tot = ET.fromstring(raw).find("os:totalResults", NS)
    time.sleep(3)
    return int(tot.text) if tot is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("term")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lead", type=int, default=9, help="decision point = T0 + lead months")
    a = ap.parse_args()

    sys.path.insert(0, str(MB))
    from marketbrain.eval.breakout_signature import theme_features, SIGNATURE_FEATURES
    from sklearn.ensemble import HistGradientBoostingClassifier

    print(f"[score] fetching arXiv footprint for '{a.term}'...", flush=True)
    recs = pd.DataFrame(fetch_papers(a.term))
    if recs.empty:
        print("[score] no arXiv papers found -> not a research theme (yet)."); return
    recs["record_id"] = recs["record_id"].astype(str)

    # (fix 1) keep only AI/ML-core-category papers -> drops off-topic homonyms (a q-bio paper that merely
    # uses the phrase), the main cause of a wrong T0.
    core = set(CATS)
    recs = recs[recs["categories"].apply(lambda cs: bool(set(cs) & core))].reset_index(drop=True)
    if recs.empty:
        print("[score] no core-category papers -> not an AI/ML research theme (may be industry/GitHub-native)."); return

    # denominator over the theme's full life (cached across terms -- identical for every term)
    first, latest = min(recs["month"]), max(recs["month"])
    grid = month_iter(first, latest)
    denom = json.loads(DENOM_CACHE.read_text()) if DENOM_CACHE.exists() else {}
    need = [m for m in grid if m not in denom]
    print(f"[score] denominator: {len(grid)} months ({len(need)} to fetch, {len(grid)-len(need)} cached)...", flush=True)
    for m in need:
        denom[m] = cat_count(m)
        DENOM_CACHE.write_text(json.dumps(denom))                # persist incrementally (survives a 429 abort)
    total_by_month = pd.Series({m: denom[m] for m in grid}, dtype=float)

    # (fix 2) detect T0 with the SAME emergence detector the corpus uses (birth-month of the surge), not a
    # crude gap rule -> isolates the real breakout even amid pre-existing sparse usage (e.g. KAN's 2024 spike).
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    nth = recs.groupby("month")["record_id"].nunique().reindex(grid, fill_value=0)
    dv = total_by_month.to_numpy()
    share = pd.Series(np.divide(nth.to_numpy(dtype=float), dv, out=np.zeros(len(grid)), where=dv > 0), index=grid)
    res = durable_breakout_label(share, nth.astype(int), EmergenceLabelConfig(), corpus_end=latest)
    t0 = res.get("T0")
    if t0 is None or t0 not in grid or grid.index(t0) + a.lead >= len(grid):
        print(f"[score] no matured breakout for '{a.term}' (T0={t0}) -> too early / too small to score."); return
    t_dec = grid[grid.index(t0) + a.lead]
    recs = recs[recs["month"] >= t0].reset_index(drop=True)       # keep the emergence cluster only
    print(f"[score] {len(recs)} papers in cluster | T0={t0} | decision T0+{a.lead}={t_dec} | latest={latest}", flush=True)

    # features AS-OF the 9-month decision point, and AS-OF today (fuller window) -- same code the model uses
    feats_dec = theme_features(recs, t_dec, total_by_month)
    feats_now = theme_features(recs, latest, total_by_month)
    n_asof = int(recs[recs["month"] <= t_dec]["record_id"].nunique())

    # (fix 3) score with the window-NORMALIZED model (persistence_frac) so from-scratch themes are comparable
    import joblib
    bundle = joblib.load(MB / "artifacts/live_scorer/model.joblib")
    clf, FEATS = bundle["model"], bundle["features"]
    disc = pd.read_parquet(MB / "artifacts/live_scorer/train_feats_frac.parquet")
    disc = disc[disc["label"].isin(("positive", "hyped_died"))].copy()
    allp = clf.predict_proba(disc[FEATS].fillna(0).to_numpy(float))[:, 1]

    def fv(feats):
        f = dict(feats); f["persistence_frac"] = f["persistence_months"] / max(f.get("window_len", 1), 1)
        return f

    def score(feats):
        f = fv(feats); v = np.array([[f[k] for k in FEATS]], float)
        p = float(clf.predict_proba(v)[0, 1]); pct = float((allp <= p).mean() * 100)
        L = "L5" if pct >= 90 else "L4" if pct >= 70 else "L3" if pct >= 45 else "L2" if pct >= 25 else "L1"
        return {"sig_score": round(p, 3), "percentile": round(pct, 1), "rating": L}

    s_dec, s_now = score(feats_dec), score(feats_now)

    # (new) CURRENT TRAJECTORY -- observable now, no 2-year maturation needed: is the theme accelerating,
    # holding, cooling, or faded? Describes the theme's OWN activity vs its peak (no cross-theme comparison).
    tgrid = month_iter(t0, latest)
    tc = recs.groupby("month")["record_id"].nunique().reindex(tgrid, fill_value=0).astype(float)
    sm = tc.rolling(3, center=True, min_periods=1).mean()
    peak_val = float(sm.max()); peak_month = str(sm.idxmax())
    recent = float(sm.iloc[-7:-1].mean()) if len(sm) >= 7 else float(sm.iloc[-3:].mean())  # exclude the (incomplete) latest month
    retention = recent / peak_val if peak_val > 0 else 0.0
    msp = len(month_iter(peak_month, latest)) - 1
    tail = sm.iloc[-12:].to_numpy()
    slope = float(np.polyfit(range(len(tail)), tail, 1)[0] / (tail.mean() + 1e-9)) if len(tail) >= 3 else 0.0
    # combine current-vs-peak (retention), how long ago the peak was (msp), and recent direction (slope)
    if retention < 0.30:
        state = "faded"                                   # collapsed well below peak
    elif msp <= 6 and slope > 0.03:
        state = "accelerating"                            # peaked recently and still climbing
    elif (msp >= 12 and slope < 0) or slope < -0.04:
        state = "cooling"                                 # peaked long ago and declining, OR clearly falling now
    else:
        state = "sustained"                               # at/near a recent peak, holding
    traj = {"state": state, "retention": round(retention, 2), "peak_month": peak_month,
            "months_since_peak": msp, "recent_slope": round(slope, 3)}
    print(f"[score] trajectory: {state} (retention {retention:.2f} vs peak {peak_month}, {msp}mo ago, slope {slope:+.3f})", flush=True)
    # per-feature context: new-theme value vs durable / fad medians (normalized feature set)
    med_pos = disc[disc["label"] == "positive"][FEATS].median()
    med_fad = disc[disc["label"] == "hyped_died"][FEATS].median()
    fdec = fv(feats_dec)
    ctx = {k: {"value": round(float(fdec[k]), 3), "durable_median": round(float(med_pos[k]), 3),
               "fad_median": round(float(med_fad[k]), 3)} for k in FEATS}

    result = {"term": a.term, "t0": t0, "t_dec": t_dec, "lead_months": a.lead, "latest": latest,
              "n_papers_asof_decision": n_asof, "n_papers_now": int(recs["record_id"].nunique()),
              "asof_decision": s_dec, "asof_today": s_now, "trajectory": traj, "features_context": ctx,
              "note": "Real out-of-sample forward call: this theme emerged after the corpus ends (2023-02). "
                      "Features computed with the same theme_features() the trained model uses; core-category "
                      "denominator fetched live from arXiv."}
    print(json.dumps({k: result[k] for k in ("term", "t0", "t_dec", "n_papers_asof_decision", "n_papers_now",
                                             "asof_decision", "asof_today")}, indent=1), flush=True)
    if a.out:
        __import__("pathlib").Path(a.out).write_text(json.dumps(result, indent=1))
        print(f"[score] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
