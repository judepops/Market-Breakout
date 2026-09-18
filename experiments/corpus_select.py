"""Data-driven corpus selection: which categories HOST the discriminative themes (signal) vs merely inflate the
share-denominator (dilution)?

    python -m experiments.corpus_select

For each arXiv primary category, compares its share of the denominator against how many discriminative themes
(and durable-breakout POSITIVES) call it home (modal primary_category of the theme's <=T_dec papers). Categories
that are large in the denominator but host ~zero breakouts are pure diluters; the recommended eval corpus is the
set of categories that host ~all the positives. Chosen from evidence, not hand-picked.
"""
from __future__ import annotations
import json
import pathlib


def main():
    import pandas as pd
    from marketbrain.eval.dataset import build_discriminative_dataset

    print("[select] building discriminative dataset (all 28 cats)...", flush=True)
    d = build_discriminative_dataset(None)
    disc, records, t2r = d["disc"], d["records"], d["term_to_records"]
    tdec = disc.set_index("theme_id")["T_dec"]
    idx = set(records.index)
    doc_by_cat = records["primary_category"].value_counts()
    n_docs = len(records)

    home = {}
    for t in disc["theme_id"]:
        recs = records.loc[list(t2r.get(t, set()) & idx)]
        recs = recs[recs["month"] <= tdec[t]]
        if len(recs):
            home[t] = recs["primary_category"].mode().iloc[0]
    disc = disc.assign(home=disc["theme_id"].map(home))
    pos = disc[disc["label"] == "positive"].groupby("home").size()
    alld = disc.groupby("home").size()
    tot_pos = int((disc["label"] == "positive").sum())

    rows = []
    for cat, docs in doc_by_cat.items():
        p, dd = int(pos.get(cat, 0)), int(alld.get(cat, 0))
        share = docs / n_docs
        rows.append({"cat": str(cat), "denom_share": share, "disc": dd, "positives": p,
                     "signal_per_pct_denom": p / (share * 100 + 1e-9)})
    tab = pd.DataFrame(rows).sort_values("positives", ascending=False)

    print(f"\n[select] {len(disc)} discriminative themes, {tot_pos} durable positives; per home category:\n")
    print(f"{'category':10s}{'denom%':>8s}{'disc':>6s}{'pos':>5s}{'pos%':>7s}{'signal/dilution':>16s}")
    for _, r in tab.head(22).iterrows():
        print(f"{r['cat']:10s}{r['denom_share']:>7.1%}{r['disc']:>6.0f}{r['positives']:>5.0f}"
              f"{(r['positives'] / max(tot_pos, 1)):>6.1%}{r['signal_per_pct_denom']:>16.1f}")

    keep = tab[tab["positives"] > 0].copy()
    keep["cum"] = keep["positives"].cumsum() / max(tot_pos, 1)
    core = keep[keep["cum"] <= 0.97]["cat"].tolist()
    if len(keep) > len(core):                              # include the one that tips over 97%
        core.append(keep.iloc[len(core)]["cat"])
    diluters = tab[(tab["denom_share"] > 0.01) & (tab["positives"] == 0)]["cat"].tolist()
    print(f"\n[select] RECOMMENDED eval corpus (hosts ~97% of positives): {sorted(core)}")
    print(f"[select] pure diluters (>1% of denominator, 0 positives): {diluters}")

    out = pathlib.Path("artifacts/corpus_select")
    out.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out / "category_contribution.parquet")
    (out / "recommended.json").write_text(json.dumps({"core": sorted(core), "diluters": diluters,
                                                       "n_disc": len(disc), "n_positives": tot_pos}, indent=2))
    print(f"[select] wrote {out}/recommended.json")


if __name__ == "__main__":
    main()
