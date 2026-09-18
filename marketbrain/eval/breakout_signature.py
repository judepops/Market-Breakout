"""The category-agnostic durable-breakout SIGNATURE -- the detector (pre-registration §4-5 + blueprint).

A per-theme feature row computed STRICTLY as-of the decision point T_dec (= T0 + 9 months): every feature uses
only records with month <= as_of (the leakage firewall is enforced INSIDE the function, defensively). NO feature
encodes WHICH arXiv category a theme lives in -- category identity is time-biased (the named DPhil trap). Only
category-AGNOSTIC shapes are used: normalized momentum + its slope, growth ratio, author breadth + inflow,
cross-domain spread (ENTROPY/count of categories, never which), cross-listing rate, venue appearance, persistence,
macro-normalised volume share, and its 2nd-derivative acceleration. Fed to gradient-boosted trees.
"""
from __future__ import annotations
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

SIGNATURE_FEATURES = [
    "escore_norm", "escore_slope", "growth_ratio", "author_breadth", "author_inflow",
    "cross_domain_spread", "cross_listing_rate", "venue_appearance", "persistence_months",
    "volume_share", "accel_2nd_deriv",
]
WINDOW_MONTHS = 24                       # cap the as-of window (theme history is usually shorter)
_TOP_VENUE = re.compile(
    r"\b(neurips|nips|icml|iclr|aaai|acl|emnlp|naacl|coling|cvpr|iccv|eccv|kdd|sigir|www|sigmod|vldb|"
    r"uai|aistats|colt|sigkdd|wsdm|recsys|interspeech|icassp|mlsys|colm)\b", re.I)


def _items(x):
    """Safely iterate a cell that may be a list/frozenset/numpy-array/None/NaN (parquet round-trips lists to arrays)."""
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:                                  # scalar NaN / non-iterable
        return []


def _entropy(items):
    if not items:
        return 0.0
    c = Counter(items)
    tot = sum(c.values())
    return float(-sum((v / tot) * math.log(v / tot) for v in c.values()))


def _window_months(theme_months, total_index, as_of):
    """Contiguous grid months from the theme's first record (capped) to as_of -- avoids leading-zero distortion."""
    grid = [m for m in total_index if m <= as_of]
    if not theme_months:
        return grid[-1:]
    start = max(theme_months[0], grid[-WINDOW_MONTHS] if len(grid) >= WINDOW_MONTHS else grid[0])
    return [m for m in grid if start <= m <= as_of]


def theme_features(recs, as_of, total_by_month):
    """recs: a theme's records (columns: record_id, month, authors, categories, comment, journal_ref).
    Returns the 11-feature dict using ONLY records with month <= as_of (defensive leakage firewall)."""
    from marketbrain.features.emergence import escore
    recs = recs[recs["month"] <= as_of]
    if recs.empty:
        return {f: 0.0 for f in SIGNATURE_FEATURES}
    theme_months = sorted(recs["month"].unique())
    months = _window_months(theme_months, list(total_by_month.index), as_of)
    mset = set(months)
    win = recs[recs["month"].isin(mset)]
    n = win["record_id"].nunique()
    series = (win.groupby("month")["record_id"].nunique().reindex(months, fill_value=0).astype(float))
    svals = series.to_numpy()

    half = max(len(months) // 2, 1)
    base, active = svals[:half].sum(), svals[half:].sum()
    growth_ratio = float(active / base) if base > 0 else float(active > 0)

    es = escore(svals.tolist())
    escore_norm = float(es / math.sqrt(max(n, 1)))
    mean = svals.mean() or 1.0
    escore_slope = float(np.polyfit(range(len(svals)), svals, 1)[0] / mean) if len(svals) >= 2 else 0.0

    last3 = months[-3:]
    early = set(m for m in months if m not in last3)
    auth_recent = {a for aset in win[win["month"].isin(last3)]["authors"] for a in _items(aset)}
    auth_early = {a for aset in win[win["month"].isin(early)]["authors"] for a in _items(aset)}
    all_auth = auth_recent | auth_early
    author_breadth = len(all_auth) / max(n, 1)
    author_inflow = len(auth_recent - auth_early) / max(len(auth_recent), 1)

    cats = [c for cl in win["categories"] for c in _items(cl)]
    cross_domain_spread = _entropy(cats)
    cross_listing_rate = float(np.mean([len(_items(cl)) >= 2 for cl in win["categories"]]))
    venue_appearance = float(np.mean([
        bool(_TOP_VENUE.search(f"{c or ''} {j or ''}")) for c, j in zip(win["comment"], win["journal_ref"])]))
    persistence_months = int((svals > 0).sum())

    total_arr = total_by_month.reindex(months).to_numpy(dtype=float)
    theme_last = series.reindex(last3, fill_value=0).sum()
    total_last = total_by_month.reindex(last3).fillna(0).sum()
    volume_share = float(theme_last / total_last) if total_last > 0 else 0.0
    share = np.divide(svals, total_arr, out=np.zeros_like(svals), where=total_arr > 0)
    accel_2nd_deriv = float(share[-1] - 2 * share[-2] + share[-3]) if len(share) >= 3 else 0.0

    return {"escore_norm": escore_norm, "escore_slope": escore_slope, "growth_ratio": growth_ratio,
            "author_breadth": author_breadth, "author_inflow": author_inflow,
            "cross_domain_spread": cross_domain_spread, "cross_listing_rate": cross_listing_rate,
            "venue_appearance": venue_appearance, "persistence_months": persistence_months,
            "volume_share": volume_share, "accel_2nd_deriv": accel_2nd_deriv}


def build_feature_matrix(themes, term_to_records, records_indexed, labels_df, total_by_month):
    """Per born theme with a T_dec, the as-of feature row + its label. themes: {theme_id: set(terms)};
    term_to_records: {term: set(record_id)}; records_indexed: corpus indexed by record_id with the record fields."""
    rows = []
    index_set = set(records_indexed.index)
    for r in labels_df.itertuples(index=False):
        t_dec = r.T_dec
        if not isinstance(t_dec, str):
            continue
        rec_ids = (set().union(*(term_to_records.get(t, set()) for t in themes[r.theme_id])) or set()) & index_set
        if not rec_ids:
            continue
        recs = records_indexed.loc[list(rec_ids)].reset_index()        # record_id back to a column
        feats = theme_features(recs, t_dec, total_by_month)
        feats.update({"theme_id": r.theme_id, "label": r.label, "T_dec": t_dec, "parent_id": r.parent_id})
        rows.append(feats)
    return pd.DataFrame(rows)


def fit_predict_oof(features_df, folds):
    """Out-of-fold P(durable) from gradient-boosted trees over the splits. folds: list of (train_df, test_df)
    on the {positive, hyped_died} subset. Returns a Series of OOF scores indexed by theme_id."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    disc = features_df[features_df["label"].isin(("positive", "hyped_died"))]
    out = {}
    for train_df, test_df in folds:
        tr = disc[disc["theme_id"].isin(train_df["theme_id"])]
        te = disc[disc["theme_id"].isin(test_df["theme_id"])]
        if tr.empty or te.empty or tr["label"].nunique() < 2:
            continue
        y = (tr["label"] == "positive").astype(int).to_numpy()
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=0)
        clf.fit(tr[SIGNATURE_FEATURES].to_numpy(), y)
        proba = clf.predict_proba(te[SIGNATURE_FEATURES].to_numpy())[:, 1]
        for tid, p in zip(te["theme_id"], proba):
            out[tid] = p
    return pd.Series(out, name="signature")
