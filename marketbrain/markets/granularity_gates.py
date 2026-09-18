"""Deterministic granularity gates -> candidate idea-phrases at the "RL with verifiable rewards" altitude.

The LLM never sees this stage; granularity is guaranteed HERE, deterministically, by five gates:
  1. doc-frequency band   -- a phrase must appear in >= MIN_DOCS papers and <= DF_CEIL of the corpus
                             (the ceiling structurally kills "language model" / "neural network").
  2. PMI-lift             -- a multi-gram's tokens must co-occur far more than chance (a real concept).
  3. generic-unigram drop -- bare boilerplate words ("model", "method") are removed.
  4. all-generic drop      -- a phrase whose every token is generic ("deep learning model") is removed
                             (the single-parent guard: nothing but a generic head + generic modifiers).
  5. noun-phrase (POS)    -- the phrase's head must be a NOUN/PROPN, removing the adjective/adverb/verb noise
                             ("materially", "survives", "auditable") that bare n-grams let through.

`gate(corpus, recent_months, prior_months)` returns the surviving candidate phrases with their stats
(doc_freq, recent/prior counts, growth) -- the input to the reason-cluster loop.
"""
from __future__ import annotations
import math
from collections import Counter

DF_CEIL = 0.08
MIN_DOCS = 25
PMI_MIN = 2.0
NOUN_POS = {"NOUN", "PROPN", "X"}        # X keeps unknown method acronyms (gepa, opd, llada)
# Audit P2: spaCy mis-tags isolated finite verbs as NOUN/PROPN (no sentence context), so a deterministic
# verb-form guard runs ALONGSIDE the POS check. Conservative by design (curated 3rd-person-singular verbs + a
# few unambiguous verb suffixes) so it never false-drops noun heads; the LLM is_market pass catches the residue.
# A full in-context majority-POS pass (tag each gram in its source sentences) is the principled upgrade.
VERB_WORDS = frozenset("""destabilizes mediates emits outperforms underperforms leverages accelerates generalizes
optimizes normalizes regularizes summarizes demonstrates evaluates investigates proposes introduces achieves
improves enables exploits mitigates facilitates formulates approximates converges diverges propagates entails
yields exhibits encompasses""".split())
VERB_SUFFIXES = ("izes", "izing", "ifies", "ified")
GENERIC = frozenset("""model models method methods approach approaches system systems framework frameworks
problem problems task tasks performance results result dataset datasets function functions analysis prediction
algorithm algorithms training network networks learning data feature features study studies application
applications technique techniques process processes design designs evaluation deep neural large scale based
novel state art using via toward towards paper work approach setting settings""".split())


def term_counts(corpus, recent_months, prior_months, *, ngram_max=3):
    """Per-phrase doc-frequency + recent/prior window counts over the corpus (title + abstract)."""
    from marketbrain.signals.arxiv_records import _terms_in
    df_count, recent, prior = Counter(), Counter(), Counter()
    rset, pset = set(recent_months), set(prior_months)
    for row in corpus.itertuples(index=False):
        terms = _terms_in(f"{row.title}. {row.abstract}", ngram_max=ngram_max)
        in_r, in_p = row.month in rset, row.month in pset
        for t in terms:
            df_count[t] += 1
            if in_r:
                recent[t] += 1
            elif in_p:
                prior[t] += 1
    return df_count, recent, prior, len(corpus)


def _pmi_ok(term, df_count, n):
    ws = term.split()
    if len(ws) == 1:
        return True
    prod = 1.0
    for w in ws:
        dw = df_count.get(w, 0)
        if dw == 0:
            return False
        prod *= dw / n
    return math.log((df_count[term] / n) / prod) >= PMI_MIN * (len(ws) - 1)


def _looks_like_verb(tok):
    """Deterministic finite-verb test for a single head token (audit P2). High precision, no noun false-drops."""
    return tok in VERB_WORDS or (len(tok) > 6 and tok.endswith(VERB_SUFFIXES))


def _is_fragment(phrase):
    """Reject tokenization artifacts ('b--', 'b-a', 'gpt-oss-'): a hyphen-token with an empty or all-single-char
    part. Keeps real hyphenated method names (k-means, t-sne, q-learning, r-cnn) where one part is a real word."""
    for tok in phrase.split():
        if "-" in tok:
            parts = tok.split("-")
            if any(p == "" for p in parts) or all(len(p) <= 1 for p in parts):
                return True
    return False


def _noun_headed(phrases):
    """Keep phrases whose head (last token) is a NOUN/PROPN/X and not a finite verb -- drops adj/adv/verb noise."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    keep = set()
    for phrase, doc in zip(phrases, nlp.pipe(phrases, batch_size=1000)):
        toks = [t for t in doc if not t.is_punct and not t.is_space]
        if toks and toks[-1].pos_ in NOUN_POS and not _looks_like_verb(toks[-1].text.lower()):
            keep.add(phrase)
    return keep


def share_growth(recent, prior, n_recent, n_prior):
    """Growth as a ratio of per-window document SHARES, not raw counts (audit P4): a flat-share theme reads ~1x
    even as the corpus inflates. +1 Laplace on counts; window sizes guard the denominators."""
    return ((recent + 1) / (n_recent + 1)) / ((prior + 1) / (n_prior + 1))


def gate(corpus, recent_months, prior_months, *, ngram_max=3, pos_filter=True):
    """Run all gates. Returns a DataFrame [phrase, doc_freq, recent, prior, growth] sorted by growth (share-based)."""
    import pandas as pd
    df_count, recent, prior, n = term_counts(corpus, recent_months, prior_months, ngram_max=ngram_max)
    rset, pset = set(recent_months), set(prior_months)
    n_recent = int(corpus["month"].isin(rset).sum())
    n_prior = int(corpus["month"].isin(pset).sum())
    ceil = DF_CEIL * n
    kept = []
    for t, c in df_count.items():
        if c < MIN_DOCS or c > ceil:
            continue
        if _is_fragment(t):                                    # tokenization-artifact guard (b--, b-a, gpt-oss-)
            continue
        ws = t.split()
        if len(ws) == 1 and t in GENERIC:
            continue
        if len(ws) > 1 and all(w in GENERIC for w in ws):      # all-generic / single-parent guard
            continue
        if not _pmi_ok(t, df_count, n):
            continue
        kept.append(t)
    if pos_filter:
        kept = list(_noun_headed(kept))
    rows = [(t, df_count[t], recent[t], prior[t], share_growth(recent[t], prior[t], n_recent, n_prior)) for t in kept]
    return pd.DataFrame(rows, columns=["phrase", "doc_freq", "recent", "prior", "growth"]) \
        .sort_values("growth", ascending=False).reset_index(drop=True)
