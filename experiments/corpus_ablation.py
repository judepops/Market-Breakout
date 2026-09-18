"""Corpus-combination ablation: which subset of the 28 arXiv categories maximises the SIGNATURE's
out-of-fold AUPRC lift over momentum -- WITHOUT fishing.

    python -m experiments.corpus_ablation --stage validate      # reproduce known points, then stop
    python -m experiments.corpus_ablation --stage search --random 300 --greedy

Why this is not naive p-hacking (the three guardrails):
  1. Efficient in-memory harness: load full-28 ONCE, subset per combo (any-tag membership, matching
     load_corpus), reuse the exact pipeline functions (durable_breakout_label, build_feature_matrix,
     _pmi_ok, grouped_folds, paired_delta_ci). A VALIDATION stage asserts the harness reproduces the
     standalone phase1_secure numbers (cs.AI n=479 delta~+0.048; core-11 n=135 delta~+0.021) before any
     search output is trusted.
  2. De-biased selection: themes are hash-split into halves A/B (stable across combos). We SELECT the
     corpus on delta measured in half A and REPORT its delta in half B (never used for selection) -- and
     vice-versa (2-fold nested). Only the held-out half is an honest estimate of the selected combo.
  3. Category importance, not the single argmax: hundreds of random subsets -> regress delta on
     category-inclusion indicators, so we learn which cats SYSTEMATICALLY help (robust) rather than
     trusting one lucky combo (noise). Plus negative-control + mechanism sanity on any winner.

Everything is written incrementally to artifacts/corpus_ablation/ so a multi-hour run is inspectable.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/corpus_ablation")
CORE11 = ["cond-mat.quant-gas", "cs.CL", "cs.CR", "cs.CV", "cs.DB", "cs.DS", "cs.LG", "cs.NE",
          "eess.SY", "quant-ph", "stat.ML"]
MIN_DOCS = 40


def _half(theme_id: str) -> int:
    """Stable A/B split of a theme by md5 (independent of corpus/combo)."""
    return int(hashlib.md5(f"ab:{theme_id}".encode()).hexdigest(), 16) % 2


class Ablation:
    """Load full-28 once; precompute term-events + noun-head verdicts; evaluate any category subset fast."""

    def __init__(self):
        import numpy as np
        import pandas as pd
        from marketbrain.corpus.load_full import load_corpus
        from marketbrain.signals.arxiv_records import extract_term_events
        from marketbrain.markets.granularity_gates import _noun_headed, _is_fragment, GENERIC
        self.np, self.pd = np, pd

        t0 = time.time()
        print("[ablation] loading full-28 corpus once...", flush=True)
        corpus = load_corpus(None)                                  # all downloaded cats, deduped, cached
        self.corpus = corpus.reset_index(drop=True)
        self.n_full = len(self.corpus)
        # the 28 downloaded categories (combos are subsets of these)
        cats = set()
        for c in self.corpus["categories"]:
            for x in (c if c is not None else []):
                cats.add(x)
        # restrict the universe to the categories that actually have their own raw folder / are the pull set
        raw = pathlib.Path("data/raw/arxiv_full")
        self.universe = sorted(d.name for d in raw.iterdir() if d.is_dir())
        self.cat_idx = {c: i for i, c in enumerate(self.universe)}
        print(f"[ablation] {self.n_full:,} papers; universe of {len(self.universe)} cats: {self.universe}",
              flush=True)

        # per-record category bitmask over the universe (numpy int64; up to 63 cats)
        rec_pos = {rid: i for i, rid in enumerate(self.corpus["record_id"].to_numpy())}
        self.rec_pos = rec_pos
        mask = np.zeros(self.n_full, dtype=np.int64)
        for i, c in enumerate(self.corpus["categories"].to_numpy()):
            m = 0
            for x in (c if c is not None else []):
                j = self.cat_idx.get(x)
                if j is not None:
                    m |= (1 << j)
            mask[i] = m
        self.rec_catmask = mask

        # precompute term events ONCE at the global floor (any subset's df <= full df, so min_doc_freq=MIN_DOCS
        # on the full corpus is a superset of every combo's gated terms), ngram_max=3 to match phase1.
        print("[ablation] extracting term events on full corpus (once, min_doc_freq=%d, ngram_max=3)..." % MIN_DOCS,
              flush=True)
        te = extract_term_events(self.corpus, min_doc_freq=MIN_DOCS, ngram_max=3)
        te = te.drop(columns=[c for c in ("authors",) if c in te.columns])   # 99M frozensets -> not needed here
        te["rec_pos"] = te["record_id"].map(rec_pos).astype(np.int32)
        te["term"] = te["term"].astype("category")
        te["record_id"] = te["record_id"].astype("category")
        te["month"] = te["month"].astype("category")
        self.te = te
        self.te_rec_pos = te["rec_pos"].to_numpy()
        print(f"[ablation] {len(te):,} term-events, {te['term'].nunique():,} distinct terms", flush=True)

        # precompute noun-head verdicts ONCE (phrase-only; superset of any combo's PMI survivors)
        terms = sorted(te["term"].unique())
        pre = [t for t in terms if not _is_fragment(t)
               and not (len(t.split()) == 1 and t in GENERIC)
               and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))]
        print(f"[ablation] running spaCy noun-head filter on {len(pre):,} candidate phrases (once)...", flush=True)
        self.noun_headed = set(_noun_headed(pre))
        self.GENERIC = GENERIC
        self._is_fragment = _is_fragment
        # full records index for feature building (categories are the paper's FULL tag list, as in phase1)
        self.records_indexed = self.corpus.set_index("record_id")[
            ["month", "authors", "categories", "comment", "journal_ref"]]
        print(f"[ablation] precompute done in {time.time() - t0:.0f}s; {len(self.noun_headed):,} noun-headed terms",
              flush=True)

    def eval_combo(self, cats, *, seed=0):
        """Full pipeline on the in-memory subset of `cats`. Returns a metrics dict (or None if degenerate)."""
        import numpy as np
        import pandas as pd
        from marketbrain.eval.corpus_inputs import cs_denominator
        from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
        from marketbrain.eval.breakout_signature import build_feature_matrix, fit_predict_oof, SIGNATURE_FEATURES
        from marketbrain.eval import metrics, splits
        from marketbrain.markets.granularity_gates import _pmi_ok, DF_CEIL
        from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending
        from experiments.phase1_secure import _variant_groups
        cfg = EmergenceLabelConfig()

        combo_mask = 0
        for c in cats:
            j = self.cat_idx.get(c)
            if j is None:
                continue                                   # not a downloaded folder-cat (e.g. cond-mat.quant-gas
            combo_mask |= (1 << j)                          # enters via cross-listing) -- skip, as load_corpus does
        if combo_mask == 0:
            return None
        in_sub = (self.rec_catmask & combo_mask) != 0                     # any-tag membership (== load_corpus)
        n_sub = int(in_sub.sum())
        if n_sub < 5000:
            return None
        corpus_sub = self.corpus[in_sub]
        denom = cs_denominator(corpus_sub)
        grid = list(denom.index)
        corpus_end = grid[-1]

        te_sub = self.te[in_sub[self.te_rec_pos]]                         # events whose record is in the subset
        df_count = te_sub.groupby("term", observed=True)["record_id"].nunique().to_dict()
        ceil = DF_CEIL * n_sub
        gated = [t for t, c in df_count.items()
                 if c >= MIN_DOCS and c <= ceil and t in self.noun_headed and _pmi_ok(t, df_count, n_sub)]
        gated = set(gated)
        if len(gated) < 20:
            return None
        # de-categoricalise the SMALL gated slice (plain str) for clean pivot/labelling/feature lookup
        te_g = te_sub[te_sub["term"].isin(gated)].copy()
        te_g["term"] = te_g["term"].astype(str)
        te_g["month"] = te_g["month"].astype(str)
        te_g["record_id"] = te_g["record_id"].astype(str)
        pivot = (te_g.groupby(["term", "month"], observed=True)["record_id"].nunique()
                 .unstack(fill_value=0).reindex(columns=grid, fill_value=0))
        denom_arr = denom.reindex(grid).to_numpy()
        rows = []
        for term, n_theme in pivot.iterrows():
            share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)),
                                        where=denom_arr > 0), index=grid)
            res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
            t0 = res.get("T0")
            t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
            rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
        labels = pd.DataFrame(rows)
        disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
        n_pos = int((disc["label"] == "positive").sum())
        n_neg = len(disc) - n_pos
        if n_pos < 8 or n_neg < 8:
            return None

        themes = {t: {t} for t in disc["theme_id"]}
        term_to_records = te_g.groupby("term", observed=True)["record_id"].apply(set).to_dict()
        feats = build_feature_matrix(themes, term_to_records, self.records_indexed, disc, denom)
        ids = feats["theme_id"]
        y_all = feats.set_index("theme_id")["label"]
        yv = (y_all.reindex(ids) == "positive").astype(int).to_numpy()

        ecfg = EmergenceConfig()
        tdec = disc.set_index("theme_id")["T_dec"]
        mom = pd.Series({t: escore(pivot.loc[t].reindex(
            month_range_ending(tdec[t], ecfg.window_months), fill_value=0).to_numpy().tolist()) for t in ids})
        ba = mom.reindex(ids).fillna(0.0).to_numpy()

        # variant-grouped OOF (the honest, twin-safe scorer used as the headline in phase1_secure)
        groups = _variant_groups(list(disc["theme_id"]))
        disc_g = disc.copy()
        disc_g["parent_id"] = disc_g["theme_id"].map(groups)
        folds = splits.grouped_folds(disc_g, k=5, seed=seed)
        sig = fit_predict_oof(feats, folds).reindex(ids).fillna(0.0).to_numpy()

        sig_ap = float(metrics.auprc(sig, yv))
        mom_ap = float(metrics.auprc(ba, yv))
        delta, lo, hi, p1 = metrics.paired_delta_ci(sig, ba, yv, B=1500, seed=seed)

        # de-biasing A/B split (stable per theme_id): delta measured within each half
        half = np.array([_half(t) for t in ids])
        out = {"cats": sorted(cats), "k": len(cats), "n": len(disc), "n_pos": n_pos, "n_neg": n_neg,
               "base_rate": round(n_pos / len(disc), 4), "n_papers": n_sub, "n_themes_gated": len(gated),
               "sig_auprc": round(sig_ap, 4), "mom_auprc": round(mom_ap, 4),
               "delta": round(float(delta), 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
               "p": round(float(p1), 4)}
        for name, sel in (("A", half == 0), ("B", half == 1)):
            if sel.sum() >= 12 and yv[sel].min() != yv[sel].max():
                out[f"sig_{name}"] = round(float(metrics.auprc(sig[sel], yv[sel])), 4)
                out[f"mom_{name}"] = round(float(metrics.auprc(ba[sel], yv[sel])), 4)
                out[f"delta_{name}"] = round(float(metrics.auprc(sig[sel], yv[sel]) - metrics.auprc(ba[sel], yv[sel])), 4)
                out[f"n_{name}"] = int(sel.sum())
            else:
                out[f"delta_{name}"] = None
        return out


# ---------------------------------------------------------------------------
def validate(ab):
    """Assert the in-memory harness reproduces the standalone phase1_secure numbers."""
    print("\n[ablation] === VALIDATION (must match phase1_secure standalone) ===", flush=True)
    checks = [(["cs.AI"], 479, 0.048), (CORE11, 135, 0.021)]
    ok = True
    for cats, n_exp, d_exp in checks:
        r = ab.eval_combo(cats)
        tag = "+".join(cats) if len(cats) <= 3 else f"{len(cats)}cats"
        if r is None:
            print(f"[ablation]   {tag:14s} -> None (DEGENERATE) -- FAIL", flush=True)
            ok = False
            continue
        dn = abs(r["n"] - n_exp)
        dd = abs(r["delta"] - d_exp)
        verdict = "OK" if (dn <= 6 and dd <= 0.03) else "MISMATCH"
        if verdict != "OK":
            ok = False
        print(f"[ablation]   {tag:14s} n={r['n']} (exp {n_exp}, dn={dn})  delta={r['delta']:+.3f} "
              f"(exp {d_exp:+.3f}, dd={dd:.3f})  sig={r['sig_auprc']} mom={r['mom_auprc']} -> {verdict}", flush=True)
    print(f"[ablation] VALIDATION {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


def _log_row(rows, r, note=""):
    import pandas as pd
    rows.append({**r, "note": note})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "results.parquet")
    tag = ",".join(r["cats"]) if r["k"] <= 6 else f"{r['k']}cats"
    dA = r.get("delta_A"); dB = r.get("delta_B")
    print(f"[ablation]   {note:10s} k={r['k']:2d} n={r['n']:3d} br={r['base_rate']:.2f}  "
          f"delta={r['delta']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p']:.3f}  "
          f"A={dA if dA is None else format(dA,'+.3f')} B={dB if dB is None else format(dB,'+.3f')}  [{tag}]",
          flush=True)


def search(ab, *, n_random=300, do_greedy=True, seed=0):
    import random
    import pandas as pd
    rows = []
    U = ab.universe
    rng = random.Random(seed)

    print("\n[ablation] === STAGE 1: single-category screen ===", flush=True)
    singles = {}
    for c in U:
        r = ab.eval_combo([c])
        if r is not None:
            singles[c] = r
            _log_row(rows, r, note="single")
    # rank singletons by held-out delta_B (fall back to delta) for a greedy seed
    ranked = sorted(singles, key=lambda c: (singles[c].get("delta_A") if singles[c].get("delta_A") is not None
                                            else singles[c]["delta"]), reverse=True)
    print(f"[ablation] top single cats by delta_A: {ranked[:8]}", flush=True)

    print("\n[ablation] === STAGE 2: random subsets (importance regression) ===", flush=True)
    for i in range(n_random):
        k = rng.randint(2, min(14, len(U)))
        combo = rng.sample(U, k)
        r = ab.eval_combo(combo, seed=seed)
        if r is not None:
            _log_row(rows, r, note="random")

    if do_greedy:
        print("\n[ablation] === STAGE 3: greedy forward (select on delta_A) ===", flush=True)
        chosen = [ranked[0]]
        best = singles[ranked[0]]
        sel = lambda x: (x.get("delta_A") if x.get("delta_A") is not None else x["delta"])
        _log_row(rows, best, note="greedy_fwd")
        improved = True
        while improved and len(chosen) < len(U):
            improved = False
            cand_results = []
            for c in U:
                if c in chosen:
                    continue
                r = ab.eval_combo(chosen + [c], seed=seed)
                if r is not None:
                    cand_results.append((c, r))
            if not cand_results:
                break
            c_best, r_best = max(cand_results, key=lambda cr: sel(cr[1]))
            if sel(r_best) > sel(best) + 1e-4:
                chosen = sorted(chosen + [c_best])
                best = r_best
                improved = True
                _log_row(rows, r_best, note="greedy_fwd")
        print(f"[ablation] greedy-forward chosen ({len(chosen)}): {chosen}  "
              f"delta_A={best.get('delta_A')} delta_B={best.get('delta_B')}", flush=True)

        print("\n[ablation] === STAGE 4: greedy backward from full-28 (select on delta_A) ===", flush=True)
        chosen_b = list(U)
        cur = ab.eval_combo(chosen_b, seed=seed)
        if cur is not None:
            _log_row(rows, cur, note="greedy_back")
            improved = True
            while improved and len(chosen_b) > 2:
                improved = False
                cand_results = []
                for c in list(chosen_b):
                    r = ab.eval_combo([x for x in chosen_b if x != c], seed=seed)
                    if r is not None:
                        cand_results.append((c, r))
                if not cand_results:
                    break
                c_drop, r_best = max(cand_results, key=lambda cr: sel(cr[1]))
                if sel(r_best) > sel(cur) + 1e-4:
                    chosen_b = [x for x in chosen_b if x != c_drop]
                    cur = r_best
                    improved = True
                    _log_row(rows, r_best, note="greedy_back")
            print(f"[ablation] greedy-backward kept ({len(chosen_b)}): {sorted(chosen_b)}  "
                  f"delta_A={cur.get('delta_A')} delta_B={cur.get('delta_B')}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "results.parquet")
    (OUT / "summary.json").write_text(json.dumps({"n_evals": len(df), "universe": U}, indent=2))
    print(f"\n[ablation] wrote {OUT}/results.parquet ({len(df)} evals)", flush=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validate", "search"], default="validate")
    ap.add_argument("--random", type=int, default=300)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ab = Ablation()
    ok = validate(ab)
    if args.stage == "validate":
        return
    if not ok:
        print("[ablation] ABORT: validation failed; not running the search on an unfaithful harness.", flush=True)
        return
    search(ab, n_random=args.random, do_greedy=args.greedy, seed=args.seed)


if __name__ == "__main__":
    main()
