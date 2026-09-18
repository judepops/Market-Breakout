"""Co-occurrence clustering of gated candidate phrases -> idea clusters.

Deterministic greedy AVERAGE-LINKAGE (audit P3): a phrase joins a cluster only if its MEAN same-paper Jaccard
to the cluster's existing members clears the floor, so a single bridge phrase can NOT chain two otherwise-
unrelated themes into one cluster (the failure mode of plain single-linkage union-find).

Scope of the signal (honest, audit P3b): same-paper co-occurrence Jaccard captures variants that appear in the
SAME abstract -- chiefly nested n-gram families ("opd" inside "on-policy distillation opd") and parenthetical
acronym-glosses ("... on-policy distillation (OPD) ..."). It does NOT merge arbitrary mutually-exclusive surface
synonyms (authors use one form per paper, so true synonyms co-occur least). Distributional / string-similarity
merging is the upgrade for that case; for now the LLM naming stage groups what this leaves split.
"""
from __future__ import annotations
from collections import Counter

JACCARD_MIN = 0.30
MIN_COOC = 4


def cooccurrence(paper_phrase_sets):
    """paper_phrase_sets: iterable of frozenset(phrases) -> (cooc Counter{(a,b):n}, df Counter{phrase:n})."""
    cooc, df = Counter(), Counter()
    for terms in paper_phrase_sets:
        ts = sorted(terms)
        for t in ts:
            df[t] += 1
        for i in range(len(ts)):
            a = ts[i]
            for j in range(i + 1, len(ts)):
                cooc[(a, ts[j])] += 1
    return cooc, df


def cluster(phrases, cooc, df, *, jaccard_min=JACCARD_MIN, min_cooc=MIN_COOC):
    """Greedy average-linkage on qualifying co-occurring pairs -> list of clusters (lists of phrases).

    A phrase is placed in the existing cluster to which its mean Jaccard is highest, but only if that mean clears
    jaccard_min; otherwise it starts its own cluster. Deterministic (phrases processed most-frequent-first)."""
    pset = set(phrases)
    jac = {}
    for (a, b), c in cooc.items():
        if c >= min_cooc and a in pset and b in pset:
            j = c / (df[a] + df[b] - c)
            if j > 0:
                jac[(a, b)] = j
                jac[(b, a)] = j
    clusters = []
    for p in sorted(pset, key=lambda x: (-df.get(x, 0), x)):
        best_ci, best_mean = None, 0.0
        for ci, members in enumerate(clusters):
            mean = sum(jac.get((p, m), 0.0) for m in members) / len(members)   # average linkage, not single
            if mean >= jaccard_min and mean > best_mean:
                best_ci, best_mean = ci, mean
        if best_ci is None:
            clusters.append([p])
        else:
            clusters[best_ci].append(p)
    return clusters
