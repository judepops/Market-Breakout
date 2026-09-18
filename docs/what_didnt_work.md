# What did not work

A record of the routes we tried and closed, so nobody spends a month re-walking them.
Each entry is: what we tried, what happened, and why it fails. Several of these are
**pre-registered nulls**, not bugs. A null that survives hard trying is a result.

The short version: **research durability is predictable; money is not predictable from
these signals.** Everything below is the second half of that sentence, plus the feature
routes that turned out to be leaks.

---

## 1. Attention does not predict markets (the big one)

**Tried:** using public attention and adoption (Hacker News, Wikipedia, GitHub, arXiv) to
predict which markets or companies produce outliers ($500M+ raised).

The naive version looks like it works:

| naive correlation (lifetime, no time gate) | rho |
| --- | --- |
| a market's research activity vs its VC funding in the following window | +0.36 |
| a company's lifetime Hacker News attention vs total funding raised | +0.20 |
| a company's lifetime GitHub stars vs total funding raised | +0.13 |

Every one collapses once measured honestly:

- **Lead-lag (the cleanest kill).** Restrict Hacker News attention to buzz a company
  received *strictly before its first funding round*: **+0.20 falls to +0.036**. Only
  **650 of 11,783** companies (5.5%) had any pre-funding buzz at all. There is almost
  nothing there to predict from.
- **Cross-correlation, 44 markets.** Does an attention peak *lead* a funding peak? Median
  lead is **-4 months** (attention *lags* funding). The real peak correlation (0.23) is no
  better than the same test on **shuffled** data (0.21), **p = 0.398**. No lead structure.
- **Event study.** GitHub stars for $500M outliers vs the rest, aligned on the biggest
  funding round: the curves separate **only at the raise**, not in the years before.
- **Per-market, leakage-safe.** Scored one market at a time, signals sit at ~0.5 AUC.
  GitHub adoption does reach **0.60 to 0.66** in a dozen open-source-heavy data and
  developer markets, so it is not literally zero, but it is weak, patchy, and confined to
  where activity happens to be public.
- **Powered company test.** n ~ 11,800 companies, 273 themes: theme emergence at founding
  against every outcome is **|rho| <= 0.10** and sign-incoherent, age-controlled and
  cluster-bootstrapped.

**Why it fails:** funding creates the footprint. A company earns its stars, threads and
Wikipedia page *because* it raised money and hired people to build and promote. Attention
is a receipt for capital, not a forecast of it. The weak association that does exist runs
**backwards**.

**Do not:** re-run attention-to-funding with a new smoother, a new lag, or a new signal
source. The arrow points the wrong way; better measurement of the wrong arrow will not help.

## 2. Capital and market size do not predict outliers either

**Tried:** predict a market's outlier rate (share of a 3-year founding cohort reaching
$500M) from its capital intensity (median first round) and its size. Leave-one-market-out,
on the 44 markets mature enough to have resolved (formation windows 2005 to 2016).

**Result:** it **ties guessing the average rate for every market**. The model is 0.1
percentage points *worse* and wins in only **18 of 100** bootstrap resamples.

So it is not that we were missing the obvious signal. Even the money does not predict it.

## 3. Lower / continuous funding thresholds

**Tried:** the outlier threshold is $500M. We tried loosening it ($10M, $50M, $100M) and
treating funding as a continuous target, on the theory that more positives would buy
statistical power.

**Result:** no. It does not work at any threshold. The problem is not the rarity of the
label, it is that the signals do not carry the information.

**Note:** this was raised again in the 2026-07-14 meeting. It is already a tested null.

## 4. Durable research does not buy a durable market

**Tried:** if we can predict durable research (we can, +0.23 AUPRC over momentum), does a
durable theme map to a better market? Bridged themes onto VMQ's outcome-labelled markets
and tested every rung: capital, outcomes, timing.

**Result:** flat across **nine pre-registered nulls**. VMQ market rating: null. Market
size: null. Funding dynamics: null. $100M+ exit rate: **70% vs 73%** (fads marginally
ahead, well inside noise).

Only about **1 in 5** durable themes ever produces a funded company (~20% vs a ~16% base
rate), which is a lower bound given the bridge is incomplete.

**Read it as:** "no durable advantage" (a clean null), **not** "fads make better markets".
Research durability tells you the science is real. It does not tell you the market is.

## 5. LLM promise rating

**Tried:** ask an LLM to score each idea's promise 0 to 1, combine with momentum, see if it
predicts durability.

**Result:** the structural detector beats momentum-plus-LLM by **+0.14**. The LLM rates
almost every idea around **0.85**. It is optimism, not discrimination.

## 6. The thesis engine (blind-graded)

**Tried:** an LLM that reasons from an idea to its markets and candidate companies. Graded
blind to avoid hindsight: one model stripped each idea's early papers into an anonymised,
paraphrased dossier, a second wrote the thesis from that dossier alone, and a third,
different model scored it without seeing the outcome.

**Result:** across 80 forced choices it picked the durable idea **42.5%** of the time. A
coin flip. Killed as a predictor (it may still be useful as a writing aid).

## 7. Broadening the corpus to admit newer themes

**Tried:** the labelled radar stops at 2023. We broadened to 16 categories, lowered
`--min-docs` to 25, raised `--df-ceil` to 0.15, and ran the full corpus to **2026-06**,
expecting newer themes.

**Result:** **365 themes, newest T0 still 2023-02.** Zero themes from the last three years.

**Why:** durability labelling needs the full arc observed (roughly 3 years). Anything newer
is **censored** by construction: it grew, but we cannot yet see whether it lasts. Also,
adding categories **hurts** — emergence is a theme's *share* of the corpus, so diluting the
denominator suppresses detection (202 themes gained from new categories, 220 lost, net -18).

**Implication:** the labelled radar is inherently a <=2023 artifact. Newer themes can only
ever be **predictions** (live / newborn cohorts), never validated labels. That is not a
limitation to engineer around; it is what durability means.

## 8. Institution / affiliation features (a leak, not a feature)

**Tried:** author affiliation and institution prestige ("if MIT and Oxford publish on it at
the same time, it is durable, earlier"), sourced from OpenAlex.

**Result:** killed before building. OpenAlex institution coverage on our arXiv DOIs:

| era | papers with >= 1 institution |
| --- | --- |
| 2016-18 | 39% |
| 2021-22 | 13% |
| **2025-26** | **0%** |

Only 10% of authors carry an affiliation at all. Mamba and KAN both return authors with
zero institutions.

**Why it fails, and this is the important part:** OpenAlex takes institutions from the
*published* version, so an affiliation only appears once a paper has been **accepted**.
Institution presence is a **proxy for publication success**, i.e. the outcome leaking
backwards. That era gradient (39% -> 13% -> 0%) *is* the leak signature. It is the same
leak as `venue_appearance` wearing a different field name.

**Corollary:** any affiliation route through published metadata is leaky by construction,
and `author.last_known_institution` is anachronistic (it is current affiliation, not
as-of). In-corpus author features (`marketbrain/eval/author_features.py`) are the safe
version: prior papers, newcomer ratio, star, prior hit, all computed as-of.

## 9. Gotcha: `venue_appearance` is leaky

Not our finding (the team's), recorded because it bit the headline. `venue_appearance` reads
`comment` / `journal_ref`, which authors **backfill after acceptance** (~4 month median
lag). It was look-ahead bias, and it was the model's single strongest feature. Dropped from
the leak-free concept model. See `experiments/venue_sensitivity.py`.

## 10. Method note: do not validate trajectory against the durability label

**Tried:** checking whether the trajectory read-out (accelerating / sustained / cooling /
faded) "agrees" with the durable/fad label.

**Result:** it does not, and that is correct. They are **orthogonal by design**. Trajectory
is what a theme is doing *now*; the label is whether it proved durable *then*. A theme that
was durable in 2018 is often cooling by 2026. Agreement is the wrong test. Judge trajectory
on its own terms.

---

## The one-line summary for whoever inherits this

Predicting **research durability** works and transfers across sciences. Predicting **money**
from public signals does not, because those signals are downstream of the money. Do not
spend your internship on the second one. We already did.
