"""Export the durability radar JSON (schema 2) for market-brain-app.

    python -m experiments.build_radar_data --out ../market-brain-app/public/mb-data.json \
        [--cats cs.AI] [--min-docs 40] [--live-months 48] [--carry-old <old mb-data.json>]

Labeled cohort  : positive / hyped_died themes, features as-of T_dec (T0+9), logistic OOF probs
                  (grouped 5-fold, radar_interpretability.py pattern).
Live cohort     : censored themes born within --live-months of corpus end, features as-of the
                  corpus end month, scored by the logistic fit on all labeled rows.
Tiers           : percentile of P(durable) among labeled OOF probs -> TIER_CUTS (frozen to the
                  v1 snapshot's ~10/20/25/20/25% distribution).
Trajectory      : marketbrain.eval.trajectory.theme_trajectory as-of corpus end, both cohorts.
Series          : raw monthly distinct-record counts per theme, leading zeros trimmed.
Carry-over      : desc / market / comparables joined from the previous snapshot by theme id.
"""
from __future__ import annotations

import argparse
import json
import pathlib

# feature -> (display label, plain-language description); copied verbatim from the v1 snapshot.
DIM_META = {
    "venue_appearance":    ("Venue penetration",   "Did early papers reach strong venues (NeurIPS etc.)? A seriousness signal — the top driver."),
    "persistence_months":  ("Persistence",          "How much of its early life the idea stayed active. The strongest single durability signal."),
    "escore_slope":        ("Emergence slope",      "How steeply activity was rising at the decision point."),
    "author_breadth":      ("Author breadth",       "How many distinct authors adopted it early — broad uptake, not one lab."),
    "accel_2nd_deriv":     ("Acceleration",         "Is the growth itself speeding up, rather than plateauing?"),
    "author_inflow":       ("New-author inflow",    "Share of recent authors new to the idea — fresh, spreading adoption."),
    "volume_share":        ("Volume share",         "The idea's share of the whole field's recent output."),
    "growth_ratio":        ("Growth ratio",         "Raw growth of activity. High growth ALONE is a mild fad signature (negative in the model)."),
    "escore_norm":         ("Emergence level",      "Overall emergence magnitude, size-normalised."),
    "cross_domain_spread": ("Cross-domain spread",  "How many fields it touches. Hypothesised to matter, but refuted — near-random."),
    "cross_listing_rate":  ("Cross-listing",        "How often its papers span multiple areas. Essentially no effect."),
    # author-quality features (in-corpus, as-of — marketbrain/eval/author_features.py)
    "auth_prior_papers":   ("Team experience",      "How many earlier papers the founding authors had. The strongest author signal."),
    "auth_newcomer_ratio": ("Fresh blood",          "Share of founding authors publishing for the first time."),
    "auth_star":           ("Star author",          "Whether a highly prolific author is on the founding team."),
    "auth_prior_hit":      ("Proven winners",       "Share of founding authors who were early on a previously proven durable theme."),
}

# durability-percentile floor -> tier; reproduces the v1 distribution (~10/20/25/20/25%).
TIER_CUTS = [(90, "L5"), (70, "L4"), (45, "L3"), (25, "L2"), (0, "L1")]

# Known tokenizer shards from model/benchmark names ("Llama-3-8B-Instruct" -> "b-instruct",
# "MT-Bench" -> "bench", "GPT-4o-mini" -> "mini"). Reviewed per build; not concepts.
TERM_ARTIFACTS = frozenset(
    {"b-instruct", "bench", "backends", "ceiling", "instruct", "lite", "mini", "restores"}
)


def _norm_tokens(term: str) -> frozenset:
    """Token set with plural 's' stripped, so "vlm" and "vlms" compare equal."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in term.split())


def dedupe_cohort(theme_ids: list[str], term_to_records: dict) -> list[str]:
    """Drop near-duplicate themes within one cohort.

    A theme is dropped when another kept theme (higher document frequency, kept first)
    either contains it as a whole token ("cot" vs "chain-of-thought cot reasoning") or
    shares most of its records (Jaccard > 0.5). Keeps the higher-df spelling — the name
    people actually use.
    """
    by_df = sorted(theme_ids, key=lambda t: -len(term_to_records.get(t, set())))
    kept: list[str] = []
    for t in by_df:
        recs = term_to_records.get(t, set())
        toks = _norm_tokens(t)
        dup = False
        for k in kept:
            ktoks = _norm_tokens(k)
            if toks == ktoks:  # same concept, different spelling ("vlm" / "vlms")
                dup = True
                break
            krecs = term_to_records.get(k, set())
            inter = len(recs & krecs)
            union = len(recs | krecs) or 1
            if (toks <= ktoks or ktoks <= toks) and inter / union > 0.3:
                dup = True
                break
            if inter / union > 0.5:
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept


# Detector lead times (months after t0) for the confidence frontier. 9 is the
# pre-registered headline horizon; the earlier ones quantify what earliness costs.
HORIZONS = [1, 2, 3, 4, 6, 9]


def pick_horizon(age_months: int) -> int:
    """The largest horizon a theme's age allows (min 1 — better an early read than none)."""
    fits = [h for h in HORIZONS if h <= age_months]
    return fits[-1] if fits else HORIZONS[0]


def calibration_bins(probs, y, k: int = 5) -> list[dict]:
    """Rank-quintile calibration: OOF probs split into k equal groups (ascending),
    each with its empirical durable rate. The honest meaning of a score at this horizon."""
    import numpy as np

    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(probs)
    n = len(probs)
    bins = []
    for i in range(k):
        idx = order[i * n // k : (i + 1) * n // k]
        if len(idx) == 0:
            continue
        bins.append({
            "lo": float(probs[idx].min()),
            "rate": round(float(y[idx].mean()), 3),
            "n": int(len(idx)),
        })
    return bins


def confidence_from_bins(prob: float, bins: list[dict]) -> dict:
    """The calibration bin a probability falls in (bins ascending by lo)."""
    out = bins[0]
    for b in bins:
        if prob >= b["lo"]:
            out = b
    return out


def verdict(durable_rate: float, base_rate: float, trajectory: dict | None,
            age_months: int, decision_horizon: int) -> dict:
    """A decision, not raw numbers: does this read durable, fad, or too-early?

    ``durable_rate`` is the calibrated empirical durability of this theme's score bin
    (its probability). We turn that + the current trajectory + the theme's age into a
    plain call a VC can act on. Confidence is the calibrated rate as a percentage.
    """
    # Honesty: absolute durability is well below 50% even for the best themes (base
    # rate ~9%), so the verdict describes SIGNAL STRENGTH vs base rate, and the note
    # always states the true probability. We never call a sub-50% theme "likely durable".
    state = (trajectory or {}).get("state")
    conf = round(durable_rate * 100)
    lift = durable_rate / base_rate if base_rate else 0.0
    note = f"{conf}% of look-alikes proved durable ({lift:.0f}x base rate)"
    if age_months < decision_horizon:
        return {"call": "too_early", "label": "Too early to call", "confidence": conf,
                "note": f"only {age_months}mo of history — needs ~{decision_horizon}mo for a read"}
    if state in ("cooling", "faded"):
        return {"call": "fad_risk", "label": "Fad risk", "confidence": conf,
                "note": f"activity is {state} — {note}"}
    if lift >= 4.0:
        return {"call": "durable", "label": "Strong durable signal", "confidence": conf, "note": note}
    if lift >= 2.0:
        return {"call": "borderline", "label": "Leaning durable", "confidence": conf, "note": note}
    if lift <= 1.2:
        return {"call": "fad_risk", "label": "Fad risk", "confidence": conf, "note": note}
    return {"call": "borderline", "label": "Mixed signal", "confidence": conf, "note": note}


def tier_from_pct(pct: float) -> str:
    for floor, tier in TIER_CUTS:
        if pct >= floor:
            return tier
    return "L1"


def durability_pct(prob: float, ref_probs) -> float:
    import numpy as np

    ref = np.asarray(ref_probs, dtype=float)
    if ref.size == 0:
        return 0.0
    return round(float((ref < prob).mean() * 100), 1)


def series_payload(n_theme) -> dict | None:
    vals = n_theme.astype(int)
    nz = vals[vals > 0]
    if nz.empty:
        return None
    start = str(nz.index[0])
    monthly = [int(v) for v in vals.loc[start:]]
    return {"start": start, "monthly": monthly}


def dims_payload(feats_std: dict, coefs: dict, pct_scores: dict) -> list[dict]:
    # Only emit features the model actually uses (coefs keys) — venue is dropped in
    # the leak-free build, so it must not appear in the "why this rating" panel.
    out = []
    for f, (label, desc) in DIM_META.items():
        if f not in coefs:
            continue
        out.append({
            "label": label,
            "score": round(float(pct_scores[f]), 1),
            "contrib": round(float(coefs[f]) * float(feats_std[f]), 3),
            "desc": desc,
        })
    return out


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES, build_feature_matrix
    from marketbrain.eval.corpus_inputs import corpus_term_events, cs_denominator
    from marketbrain.eval.label import EmergenceLabelConfig, _shift_month, durable_breakout_label
    from marketbrain.eval.trajectory import _month_diff, theme_trajectory
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending
    from marketbrain.markets.granularity_gates import DF_CEIL, GENERIC, _is_fragment, _noun_headed, _pmi_ok

    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=["cs.AI"])
    ap.add_argument("--all", action="store_true", help="every downloaded category")
    ap.add_argument("--themes", choices=["ngrams", "concepts"], default="ngrams",
                    help="theme source: deterministic n-grams, or LLM concepts (concept_events.parquet)")
    ap.add_argument("--min-docs", type=int, default=40)
    ap.add_argument("--live-months", type=int, default=48)
    ap.add_argument("--newborn-months", type=int, default=12,
                    help="include not-yet-breakout themes born within this window")
    ap.add_argument("--decision-horizon", type=int, default=9,
                    help="headline decision point in months after t0 (4 for the leak-free concept model)")
    ap.add_argument("--momentum-min", type=int, default=0,
                    help="Rising-now display filter: min papers in the trailing window (0 = off)")
    ap.add_argument("--momentum-window", type=int, default=3,
                    help="trailing months for the momentum display filter")
    ap.add_argument("--fad-min-papers", type=int, default=25,
                    help="minimum total papers for a theme to appear on the Fads list")
    ap.add_argument("--carry-old", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()
    if args.themes == "concepts":
        # concepts-v1 label calibration (artifacts/concept_label_sweep.json): LLM
        # concepts count papers ABOUT an idea (~5x sparser than n-gram mentions), so
        # birth relaxes 5 -> 3-in-3mo. Growth gates stay at the pre-registered values —
        # the sweep showed loosening them destroys the signal (AUPRC 0.52 -> 0.12).
        import dataclasses
        cfg = dataclasses.replace(cfg, birth_min_records=3)

    # -- corpus -> gated themes -> labels (phase1_detector.py pattern, kept identical) --------
    corpus = load_corpus(None if args.all else args.cats, cache=args.all)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    n = len(corpus)
    if args.themes == "concepts":
        # LLM-extracted canonical concepts (experiments/concept_themes.py --build).
        # Already curated + canonicalized; only a document-frequency floor applies.
        src = pathlib.Path("data/derived/concepts/concept_events.parquet")
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run llm_concepts + concept_themes first")
        te = pd.read_parquet(src)
        te = te[te["record_id"].isin(set(corpus["record_id"]))]
        df_count = te.groupby("term")["record_id"].nunique()
        te = te[te["term"].isin(df_count[df_count >= args.min_docs].index)]
        print(f"[radar] {n:,} papers, grid {grid[0]}..{corpus_end}; "
              f"{te['term'].nunique():,} concept themes (df >= {args.min_docs})", flush=True)
    else:
        print(f"[radar] {n:,} papers, grid {grid[0]}..{corpus_end}; building gated themes...", flush=True)
        te = corpus_term_events(corpus, min_doc_freq=args.min_docs, ngram_max=3)
        df_count = te.groupby("term")["record_id"].nunique().to_dict()
        ceil = DF_CEIL * n
        gated = [t for t, c in df_count.items()
                 if c <= ceil and not _is_fragment(t)
                 and not (len(t.split()) == 1 and t in GENERIC)
                 and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))
                 and _pmi_ok(t, df_count, n)]
        gated = set(_noun_headed(gated))
        te = te[te["term"].isin(gated)]

    pivot = (te.groupby(["term", "month"])["record_id"].nunique()
             .unstack(fill_value=0).reindex(columns=grid, fill_value=0))
    denom_arr = denom.reindex(grid).to_numpy()
    rows = []
    for term, n_theme in pivot.iterrows():
        share = pd.Series(
            np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0),
            index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec,
                     "T_peak": res.get("T_peak"), "parent_id": term})
    labels = pd.DataFrame(rows)

    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
    n_pos = int((disc["label"] == "positive").sum())
    print(f"[radar] labeled cohort: {n_pos} durable / {len(disc) - n_pos} faded", flush=True)

    # live cohort, two stages:
    #   breakout — passed the growth gates, outcome window still open (label censored)
    #   newborn  — born within --newborn-months, growth still unfolding (label never);
    #              these are the "born one or two months ago" discovery surface, scored
    #              by the horizon model at whatever age they are.
    lo_month = _shift_month(corpus_end, -args.live_months)
    breakout = labels[(labels["label"] == "censored") & labels["T0"].notna() & (labels["T0"] >= lo_month)].copy()
    breakout["stage"] = "breakout"
    nb_lo = _shift_month(corpus_end, -args.newborn_months)
    newborn = labels[(labels["label"] == "never") & labels["T0"].notna() & (labels["T0"] >= nb_lo)].copy()
    newborn["stage"] = "newborn"
    live_lab = pd.concat([breakout, newborn], ignore_index=True)
    live_lab["T_dec"] = live_lab["T_dec"].fillna(corpus_end)
    print(f"[radar] live cohort: {len(breakout)} breakout (since {lo_month}) "
          f"+ {len(newborn)} newborn (since {nb_lo})", flush=True)

    # Momentum + freshness flags (not a hard filter): "hot" = >= momentum_min papers in
    # the trailing momentum_window months as-of now; "new_this_month" = the theme FIRST
    # crossed that bar in the final corpus month (the month-over-month arrivals). We score
    # every recent live theme and split Durable ideas vs Fads downstream by the verdict.
    mom_window = [_shift_month(corpus_end, -i) for i in range(args.momentum_window)]
    recent = pivot[mom_window].sum(axis=1)
    hot = set(recent[recent >= args.momentum_min].index)
    roll = pivot.T.rolling(args.momentum_window, min_periods=1).sum().T  # trailing sum per month
    crossed = roll >= args.momentum_min
    first_cross = crossed.apply(lambda r: r.idxmax() if r.any() else None, axis=1)
    new_this_month = set(first_cross[first_cross == corpus_end].index)
    total_papers = pivot.sum(axis=1)
    print(f"[radar] {len(hot)} themes hot (>= {args.momentum_min} in last {args.momentum_window}mo), "
          f"{len(new_this_month)} newly crossed in {corpus_end}", flush=True)

    # -- features -----------------------------------------------------------------------------
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()

    # theme-quality pass within each cohort: drop tokenizer artifacts + near-duplicates
    disc = disc[~disc["theme_id"].isin(TERM_ARTIFACTS)]
    live_lab = live_lab[~live_lab["theme_id"].isin(TERM_ARTIFACTS)]
    disc = disc[disc["theme_id"].isin(dedupe_cohort(list(disc["theme_id"]), term_to_records))].reset_index(drop=True)
    live_lab = live_lab[live_lab["theme_id"].isin(dedupe_cohort(list(live_lab["theme_id"]), term_to_records))].reset_index(drop=True)
    n_pos = int((disc["label"] == "positive").sum())
    print(f"[radar] after artifact/dup pass: {len(disc)} labeled ({n_pos} durable), {len(live_lab)} live", flush=True)
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref", "title"]]

    # author-quality features (in-corpus, as-of; see marketbrain/eval/author_features.py)
    from marketbrain.eval.author_features import (
        AUTHOR_FEATURES, build_author_index, early_authors, settled_hit_index,
        theme_author_features)
    from marketbrain.eval.researchers import theme_key_papers, theme_researchers

    # web-enrichment (profile link + founder flag) from experiments/enrich_researchers.py, if run
    enrich_path = pathlib.Path("data/derived/researchers/enrichment.json")
    enrichment = json.loads(enrich_path.read_text()) if enrich_path.exists() else {}
    if enrichment:
        print(f"[radar] joining web-enrichment for {len(enrichment)} researchers", flush=True)
    # Leak-free feature set: venue_appearance reads comment/journal_ref, which authors
    # backfill AFTER acceptance (~4mo median lag) — look-ahead bias, so it is dropped.
    # See experiments/venue_sensitivity.py. Clean = 10 activity signals + 4 author signals.
    FEATURES = ([f for f in SIGNATURE_FEATURES if f != "venue_appearance"] + AUTHOR_FEATURES
                if args.themes == "concepts" else SIGNATURE_FEATURES + AUTHOR_FEATURES)
    author_index = build_author_index(corpus)
    index_set = set(records_indexed.index)

    def theme_recs(tid):
        ids = term_to_records.get(tid, set()) & index_set
        return records_indexed.loc[list(ids)]

    hits = settled_hit_index([
        (r.theme_id, _shift_month(r.T_peak or r.T0, cfg.persist_hi),
         early_authors(theme_recs(r.theme_id), r.T0, as_of=corpus_end))
        for r in disc[disc["label"] == "positive"].itertuples(index=False)
    ])
    print(f"[radar] author index: {len(author_index):,} authors, "
          f"{len(hits):,} with settled durable track records", flush=True)

    def feats_at(labels_frame, horizon):
        """Feature rows for a cohort with the as-of point moved to t0 + horizon months."""
        lf = labels_frame.copy()
        lf["T_dec"] = lf["T0"].map(lambda t0: _shift_month(t0, horizon))
        out = build_feature_matrix({t: {t} for t in lf["theme_id"]}, term_to_records, records_indexed, lf, denom)
        if not len(out):
            return out
        out = out.merge(lf[["theme_id", "T0"]], on="theme_id")
        arows = [{"theme_id": r.theme_id,
                  **theme_author_features(
                      early_authors(theme_recs(r.theme_id), r.T0, as_of=r.T_dec),
                      r.T0, author_index, hits)}
                 for r in out.itertuples(index=False)]
        return out.merge(pd.DataFrame(arows), on="theme_id")

    # -- the confidence frontier: one logistic per horizon, OOF-calibrated ---------------------
    # Era-matched percentile reference per horizon: a 2024 theme's early footprint is only
    # comparable to recent-era peers, not the whole 2012+ history. Falls back when thin.
    era_floor = _shift_month(corpus_end, -72)
    frontier: dict[int, dict] = {}
    for h in HORIZONS:
        fh = feats_at(disc, h)
        Xh = fh[FEATURES].fillna(0).to_numpy(float)
        yh = (fh["label"] == "positive").astype(int).to_numpy()
        scaler_h = StandardScaler().fit(Xh)
        Xsh = scaler_h.transform(Xh)
        oof_h = np.zeros(len(fh))
        for tr_df, te_df in splits.grouped_folds(fh, k=5):
            tri = fh["theme_id"].isin(tr_df["theme_id"]).to_numpy()
            tei = fh["theme_id"].isin(te_df["theme_id"]).to_numpy()
            if len(np.unique(yh[tri])) < 2:
                continue
            m = LogisticRegression(max_iter=1000).fit(Xsh[tri], yh[tri])
            oof_h[tei] = m.predict_proba(Xsh[tei])[:, 1]
        model_h = LogisticRegression(max_iter=1000).fit(Xsh, yh)
        era_mask_h = (fh["T0"] >= era_floor).to_numpy()
        frontier[h] = {
            "feats": fh,
            "y": yh,
            "scaler": scaler_h,
            "model": model_h,
            "coefs": dict(zip(FEATURES, model_h.coef_[0])),
            "oof": oof_h,
            "auprc": float(metrics.auprc(oof_h, fh["label"].to_numpy())),
            "base_rate": round(float(yh.mean()), 3),
            "bins": calibration_bins(oof_h, yh, k=5),
            "era_ref": oof_h[era_mask_h] if int(era_mask_h.sum()) >= 40 else oof_h,
        }
        print(f"[radar] horizon t0+{h}mo: OOF AUPRC={frontier[h]['auprc']:.3f} "
              f"(base rate {frontier[h]['base_rate']:.3f})", flush=True)

    # Headline horizon = the business decision point (default 4mo for the leak-free
    # concept model; 9mo pre-registered for n-grams). Drives labeled entries + the
    # momentum comparison + meta.detector.
    HEAD = args.decision_horizon
    H = frontier[HEAD]
    feats, y, oof, scaler, model = H["feats"], H["y"], H["oof"], H["scaler"], H["model"]
    coefs = H["coefs"]

    # live cohort: score each theme at the largest horizon its age allows
    live_scored: list[tuple[dict, float, int]] = []  # (feature row, prob, horizon)
    if len(live_lab):
        live_lab = live_lab.assign(
            _h=live_lab["T0"].map(lambda t0: pick_horizon(_month_diff(t0, corpus_end))))
        for h, grp in live_lab.groupby("_h"):
            fh = feats_at(grp.drop(columns=["_h"]), int(h))
            if not len(fh):
                continue
            fh = fh.merge(grp[["theme_id", "stage"]], on="theme_id")
            Xl = frontier[int(h)]["scaler"].transform(fh[FEATURES].fillna(0).to_numpy(float))
            probs = frontier[int(h)]["model"].predict_proba(Xl)[:, 1]
            live_scored.extend((r._asdict(), float(p), int(h)) for r, p in zip(fh.itertuples(index=False), probs))

    # -- detector meta: logistic OOF vs B1 momentum -------------------------------------------
    ecfg = EmergenceConfig()

    def b1(tid, t_dec):
        usage = pivot.loc[tid].reindex(month_range_ending(t_dec, ecfg.window_months), fill_value=0).to_numpy()
        return escore(usage.tolist())

    b1_scores = np.array([b1(r.theme_id, r.T_dec) for r in feats.itertuples(index=False)])
    y_labels = feats["label"].to_numpy()
    ap_sig = metrics.auprc(oof, y_labels)
    ap_b1 = metrics.auprc(b1_scores, y_labels)
    d, lo, hi, p1 = metrics.paired_delta_ci(oof, b1_scores, y_labels, B=2000, seed=0)
    print(f"[radar] logistic OOF AUPRC={ap_sig:.3f} vs momentum {ap_b1:.3f}; "
          f"delta {d:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] p={p1:.3f}", flush=True)

    # -- assembly ------------------------------------------------------------------------------
    old = {}
    if args.carry_old and pathlib.Path(args.carry_old).exists():
        prev = json.loads(pathlib.Path(args.carry_old).read_text())
        old = {t["id"]: t for t in prev.get("themes", [])}

    # stage-3 hierarchy (concept mode): theme -> parent cluster name, if built
    clusters = {}
    cl_path = pathlib.Path("data/derived/concepts/concept_clusters.parquet")
    if args.themes == "concepts" and cl_path.exists():
        cdf = pd.read_parquet(cl_path)
        clusters = dict(zip(cdf["term"], cdf["cluster"]))

    def build_entry(fr: dict, prob: float, live_theme: bool, h: int = 9) -> dict:
        tid = fr["theme_id"]
        F = frontier[h]
        raw = {f: (float(fr[f]) if pd.notna(fr[f]) else 0.0) for f in FEATURES}
        std = dict(zip(FEATURES,
                       F["scaler"].transform(np.array([[raw[f] for f in FEATURES]]))[0]))
        ref = F["feats"][FEATURES].fillna(0)
        pct_scores = {f: float((ref[f].to_numpy() < raw[f]).mean() * 100) for f in FEATURES}
        # Band = percentile of the calibrated durability probability among the LABELED
        # cohort (absolute). This makes the band agree with the verdict (both rise with
        # the same probability) — no more "Low quartile · Strong durable signal".
        b = confidence_from_bins(prob, F["bins"]) if live_theme else None
        rank_prob = b["rate"] if live_theme else prob
        pct = durability_pct(rank_prob, F["oof"])
        n_series = pivot.loc[tid]
        carry = old.get(tid, {})
        traj = theme_trajectory(n_series, as_of=corpus_end)
        recs = theme_recs(tid)
        entry = {
            "id": tid,
            "rank": 0,  # assigned after sorting
            "durability": pct,
            "rating": tier_from_pct(pct),
            "sig_score": round(float(prob), 4),
            "class": "pending" if live_theme else fr["label"],
            "t0": fr["T0"],
            "desc": carry.get("desc", ""),
            "dims": dims_payload(std, F["coefs"], pct_scores),
            "comparables": carry.get("comparables", []),
            "trajectory": traj,
            "series": series_payload(n_series),
            "researchers": [{**r, **{k: v for k, v in (enrichment.get(r["name"]) or {}).items()
                                      if k in ("affiliation", "profile_url", "is_founder", "founder_company")}}
                            for r in theme_researchers(recs, fr["T0"], author_index)],
            "key_papers": theme_key_papers(recs, author_index, fr["T0"]),
        }
        if clusters.get(tid):
            entry["cluster"] = clusters[tid]
        if live_theme:
            entry["decided_by"] = _shift_month(fr["T0"], cfg.persist_hi)
            entry["confidence"] = {"horizon_months": h, "durable_rate": b["rate"], "n": b["n"]}
            entry["stage"] = fr.get("stage", "breakout")
            age = _month_diff(fr["T0"], corpus_end)
            entry["verdict"] = verdict(b["rate"], F["base_rate"], traj, age, HEAD)
            entry["hot"] = tid in hot
            entry["new_this_month"] = tid in new_this_month
            entry["total_papers"] = int(total_papers.get(tid, 0))
        return entry

    themes_out = [build_entry(r._asdict(), p, live_theme=False)
                  for r, p in zip(feats.itertuples(index=False), oof)]

    # Score every recent live theme, then split by the verdict into the three live buckets
    # that cover the decision space — will last / too soon to say / will fade:
    #   Durable ideas — hot AND the model calls it durable or leaning durable
    #   Emerging      — hot AND too young to call yet (the earliest sourcing surface)
    #   Fads          — the model calls fad risk (recent themes to avoid)
    all_live = [build_entry(fr, p, live_theme=True, h=h) for fr, p, h in live_scored]
    durable_out = [t for t in all_live if t["hot"] and t["verdict"]["call"] in ("durable", "borderline")]
    emerging_out = [t for t in all_live if t["hot"] and t["verdict"]["call"] == "too_early"]
    fads_out = [t for t in all_live
                if t["verdict"]["call"] == "fad_risk" and t["total_papers"] >= args.fad_min_papers]

    for coll in (themes_out, durable_out):
        coll.sort(key=lambda t: -t["durability"])
    emerging_out.sort(key=lambda t: (not t["new_this_month"], t["t0"] < corpus_end, -t["total_papers"]))
    fads_out.sort(key=lambda t: -t["total_papers"])  # biggest cautionary tales first
    for coll in (themes_out, durable_out, emerging_out, fads_out):
        for i, t in enumerate(coll):
            t["rank"] = i + 1
    print(f"[radar] emit: {len(durable_out)} durable, {len(emerging_out)} emerging, {len(fads_out)} fads "
          f"({sum(t['new_this_month'] for t in durable_out + emerging_out)} new this month)", flush=True)

    all_contribs = [abs(dd["contrib"])
                    for t in themes_out + durable_out + emerging_out + fads_out for dd in t["dims"]]
    max_contrib = round(float(np.percentile(all_contribs, 98)), 2) if all_contribs else 0.64

    # layman stat headline: how often a "durable"-flagged idea actually proves out,
    # vs picking at random (the base rate). Uses the headline-horizon top calibration bin.
    top_bin = max(H["bins"], key=lambda b: b["lo"])
    payload = {
        "meta": {
            "title": "Bedrock — Durable idea, or fad?",
            "as_of": corpus_end,
            "schema": 2,
            "n_papers": int(n),
            "n_ideas": int(te["term"].nunique()),
            "n_labeled": len(themes_out),
            "n_durable": n_pos,
            "first_month": grid[0],
            "hit_rate": round(top_bin["rate"], 3),
            "base_rate": round(H["base_rate"], 3),
            "decision_horizon": HEAD,
            "detector": {"delta_auprc": round(float(d), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                         "auprc": round(float(ap_sig), 3)},
            "max_contrib": max_contrib,
            "frontier": [{"horizon": h, "auprc": round(frontier[h]["auprc"], 3),
                          "base_rate": frontier[h]["base_rate"]} for h in HORIZONS],
        },
        "themes": themes_out,
        "live": durable_out,
        "emerging": emerging_out,
        "fads": fads_out,
    }

    out = pathlib.Path(args.out)
    tmp = out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload))
    if out.exists():
        out.rename(out.with_suffix(".v1.bak.json"))
    tmp.rename(out)
    print(f"[radar] wrote {out}: {len(themes_out)} labeled + {len(durable_out)} durable "
          f"+ {len(emerging_out)} emerging + {len(fads_out)} fads, as of {corpus_end}", flush=True)


if __name__ == "__main__":
    main()
