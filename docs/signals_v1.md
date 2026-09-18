# v1 deterministic signal catalogue

Grounded signal/feature definitions for the **v1** corpus — **deterministic, natively-dated sources only,
no LLMs**. Synthesized from the **Market-Brain** NotebookLM notebook and reconciled to the authoritative
pull plan in `../../Docs/data_acquisition.md`. Formulas are quoted as the sources specify; resolve bracketed
notebook citations to primary papers before any formal write-up.

**v1 scope:** the **four** signals below, wired in order. Each is computed **per market per month** and
ingested as an `Observation` with `reference_date` = the period it describes and `knowledge_date` = when the
value became public. **News/narrative indices (EPU/GPR) and corporate-filing (10-K) signals are deferred to
v1.1** — they need web-archive grounding + an LLM extractor + a human audit (see the bottom of this doc).

**v1 market:** `llm-dev-tools` (tooling for building LLM applications). All four signals work for it — that
is exactly why it is the first PoC market.

---

## 1. Google Trends — market interest (wire FIRST)
Mimics Kohns & Bhattacharjee (BSTS + Trends nowcasting, 2022). The cleanest leakage-free leading indicator.
- **Pull topics/categories, NOT raw terms** (raw terms catch spurious volume — "Jobs" labor vs "Steve
  Jobs"); e.g. *Large language model*, *API*, *Software framework*, category *Programming*.
- **Average ≥5–7 separate pulls** of each series (Google returns a slightly different 0–100 rescaling per
  query); fix geography and the query window.
- **Deseasonalize** with LOESS/STL. Transformations: monthly change, growth rate, raw index, LOESS decomp.
- **Dates:** `reference_date` = search-month; `knowledge_date` ≈ end of that month (near-real-time).
- **Gotcha:** relatively scaled per query — absolute levels aren't comparable across series; work in changes
  / standardized units with a fixed query window.

## 2. arXiv — research momentum (wire SECOND)
Mimics Rotolo et al. (2015) for the count signal; Carley et al. (2018) for the richer emergence score.
- **Simple v1 feature: monthly submission counts** matching the market's term set, filtered to relevant
  categories (cs.CL, cs.AI, cs.SE for `llm-dev-tools`). Leakage-safe — uses only that month's submissions.
- **Dates:** `reference_date` = `knowledge_date` = arXiv **submission date** (natively dated; no citation lag
  because we count submissions, not citations).

### Carley four-criterion emergence (the richer feature, built on arXiv)
A term qualifies as emergent when all four hold:
- **Novelty** — appears in **< 15%** of base-period records.
- **Persistence** — appears in **≥ 3** distinct periods **and ≥ 7** total records.
- **Community** — used by **≥ 2** non-co-authoring authors.
- **Growth** — appears **≥ 2×** as often in the active vs base period.

**Composite score** combines three trend variations (recent-usage ratio; short-term change ratio;
active-period slope); a term is emergent iff composite **≥ 1.77**. Per-actor features: raw =
Σ √(emergent-term scores) over that actor's records; normalized = raw / √(record count).

**Caveats (from `data_acquisition.md`, important):**
- The **15% ceiling and 1.77 cutoff are tuned to ~13k-record bibliometric corpora — re-tune for arXiv.**
- The **≥7-records / ≥3-periods floors break at monthly resolution** for a young market — use a **20–30-month
  sliding window** (keep the 3:7 base:active *ratio*), not 10 literal months.
- **Do NOT use citation/co-citation features** in the monthly panel — they need years to accrue and are the
  most leakage-prone (Rotolo's top warning).
- The exact growth-score *weighting* is in Garner et al. 2017 (not in our PDF set) — obtain it or define our
  own combination of the three trend components.

## 3. GitHub / OSS dependency-growth — developer adoption (wire THIRD; `llm-dev-tools`)
Mimics Brown et al. (OSS innovation signal, 2024). The best **durable** adoption signal (stars are gameable).
- Define the market as a **basket of canonical packages** (e.g. `langchain`, `llama-index`, `openai`,
  `transformers`, `vllm`, `dspy`, …).
- **Core metric = Δlog(dependents):** `ln(dependents at t+h) − ln(dependents at t)`. Use **dependents**
  (revealed preference), not stars.
- **Filters:** repos with **≥ 1 star**, **exclude forks**; package needs **≥ 5 existing dependents** before
  the measured release. Parse semver; weight **minor releases** (most reliable). Skip complexity-of-release
  (it didn't predict adoption).
- **Normalize by ecosystem (npm/PyPI/…) × package-size bin × version-series** before aggregating — the
  stratification *is* the method, not a detail.
- **Critical PIT gotcha:** dependents APIs return **today's count only**. For a clean historical series,
  **snapshot forward from now** (fine for a PoC going forward); reconstructing add-dates from each
  dependent's git history is expensive. So this signal's history is **shallow at first** — arXiv + Trends +
  funding carry the historical depth.
- **Dates:** keyed on **release date** / the dated dependent edge.

## 4. Funding announcements — capital flowing in (wire FOURTH)
Mimics the count-based growth operationalization (Rotolo); funding as a leading signal.
- **Where:** Crunchbase API (paid) or dated press/aggregators; each round carries an **`announced_on`** date.
- **How:** monthly counts and totals of funding rounds for companies in the market (# rounds, $ raised,
  # new entrants).
- **Dates:** `reference_date` ≈ round close; **`knowledge_date` = announcement date** (a round can close
  months before it is announced — key on announcement to stay leakage-safe).
- **Gotcha:** disclosure lag and reporting bias (small/undisclosed rounds missing).

---

## The central tension (design around it)
Every published emergence indicator needs **multi-year history** for persistence/community to mean anything.
At monthly cadence: **fast signals carry the early months** (arXiv counts, GitHub velocity, Trends slope,
funding); the **bibliometric emergence score stabilizes only after ~24–36 months** of base data.

## Deferred to v1.1 (NOT v1)
These need web-archive (Wayback) grounding and/or an LLM extractor + human audit — out of scope for the
deterministic v1 skeleton.

- **News narrative index (EPU/GPR).** Volume `V_i = (# articles in month t matching the keyword pattern) /
  (total articles same source-month)`. EPU "triple" (≥1 term from each of {economic},{uncertainty},{policy})
  + GPR grouped Booleans + negative keywords. Standardize per-source to unit-SD, average, rescale base = 100
  (GPR's simpler share method is the default). **Validation (make-or-break):** a parallel human-coded index
  with correlation **≥ 0.8** (EPU 0.86, GPR 0.84), ~87% precision on selected articles, and a **"drop
  outcome terms"** variant (remove funding/valuation/IPO) that must still correlate ~0.99 — the strongest
  proof the feature isn't contaminated by the label.
- **Corporate-filing / 10-K risk.** Policy-risk share = (# Risk-Factors sentences with policy terms) /
  (total sentences); monthly count of daily stock moves > |2.5%| attributed to policy news. Macro context:
  UMCI, PMI.

## Validation target (deterministic, GPR-style)
Report two credibility numbers for the corpus: **parallel-human-index correlation** and **false-positive
rate** on a stratified human audit (see `DATASHEET.md` §8).
