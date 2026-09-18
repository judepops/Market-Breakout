"""Phase-0 baselines to beat (pre-registration §8). The detector must clear max(B1, B2).

B1  momentum        -- Garner/Carley `escore` (trend + recent-trend + slope) over the theme's own monthly
                       usage, in the window ENDING at its decision point T_dec. No persistence modelling; this
                       is the strongest "it's going up fast" signal, and the one that fires on hyped-then-died
                       negatives, so beating it = capturing persistence the momentum cannot.
B2  momentum + LLM  -- B1 plus an LLM "is this a promising area" rating, combined by logistic regression fit on
                       the TRAIN fold. Reproduces the team's team's momentum + LLM-categorisation approach.

Both score a theme strictly from <= T_dec evidence (the detector firewall, PREREGISTRATION.json detector_firewall).
"""
from __future__ import annotations

import pandas as pd


def momentum_scores(themes, term_events, labels_df, *, cfg=None):
    """B1: per-theme Garner escore over the usage window ending at T_dec (leakage-safe). -> Series[theme_id]."""
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending
    cfg = cfg or EmergenceConfig()
    out = {}
    for row in labels_df.itertuples(index=False):
        tid, t_dec = row.theme_id, row.T_dec
        if not isinstance(t_dec, str):                 # unborn theme (T_dec None/NaN) -> no momentum
            out[tid] = 0.0
            continue
        window = set(month_range_ending(t_dec, cfg.window_months))
        terms = themes[tid]
        ev = term_events[term_events["term"].isin(terms) & term_events["month"].isin(window)]
        usage = (ev.groupby("month")["record_id"].nunique()
                 .reindex(month_range_ending(t_dec, cfg.window_months), fill_value=0).astype(float))
        out[tid] = escore(usage.tolist())
    return pd.Series(out, name="B1_momentum")


_LLM_PROMPT = (
    "An emerging-technology analyst is judging research themes at a point in time. Given ONLY the theme's name "
    "and example early paper titles (no future information), rate how likely this theme is to become a large, "
    "DURABLE research area over the next 1-2 years (not a passing fad).\n\n"
    "THEME: {name}\nEARLY TITLES:\n{titles}\n\n"
    'Return ONLY JSON: {{"promise": float}}  where promise is 0.0 (fad / niche) to 1.0 (durable breakout).'
)


def llm_promise_scores(themes, titles_by_theme, *, client=None, model="gemini-2.5-pro", use_cache=True):
    """B2's LLM feature: a 0-1 "promise" rating per theme from name + early titles (cached, abstains on error)."""
    from marketbrain.markets.reason_cluster import _call_model, _load_cache, CACHE
    import hashlib
    import json
    if client is None:
        from marketbrain.labels.gemini_grounding import make_client
        client = make_client()
    cached = _load_cache() if use_cache else {}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for tid, terms in themes.items():
        name = sorted(terms)[0] if terms else tid
        titles = titles_by_theme.get(tid, [])
        key = "promise|" + model + "|" + hashlib.sha1((name + "|".join(titles[:3])).encode()).hexdigest()
        if key in cached:
            out[tid] = cached[key].get("promise", 0.0)
            continue
        prompt = _LLM_PROMPT.format(name=name, titles="\n".join(f"- {t}" for t in titles[:3]))
        d, status = _call_model(client, model, prompt)
        promise = float(d.get("promise", 0.0) or 0.0) if status == "ok" else 0.0
        out[tid] = promise
        if use_cache and status == "ok":
            with CACHE.open("a") as f:
                f.write(json.dumps({"key": key, "result": {"promise": promise}}) + "\n")
    return pd.Series(out, name="B2_llm_promise")


def momentum_plus_llm(momentum, promise, train_labels):
    """B2: logistic combine of B1 momentum + LLM promise, fit on the TRAIN fold only. -> Series[theme_id] of P(positive).

    momentum, promise: Series indexed by theme_id. train_labels: DataFrame[theme_id,label] (the train fold,
    {positive,hyped_died} only). Returns scores for ALL theme_ids in momentum's index.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    tr = train_labels[train_labels["label"].isin(("positive", "hyped_died"))]
    y = (tr["label"] == "positive").astype(int).to_numpy()

    def feat(ids):
        return np.column_stack([momentum.reindex(ids).fillna(0.0).to_numpy(),
                                promise.reindex(ids).fillna(0.0).to_numpy()])

    if len(set(y)) < 2:                                # degenerate train fold -> fall back to momentum alone
        return momentum.rename("B2_momentum_llm")
    clf = LogisticRegression(max_iter=1000).fit(feat(tr["theme_id"].tolist()), y)
    ids = momentum.index.tolist()
    proba = clf.predict_proba(feat(ids))[:, 1]
    return pd.Series(proba, index=ids, name="B2_momentum_llm")
