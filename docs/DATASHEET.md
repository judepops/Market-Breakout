# Datasheet — market-brain corpus

Datasheets-for-Datasets (Gebru et al. 2018) + Data-Statements style, **written before collecting**
(methodology Stage 0). This documents intent, sourcing, the label, and QC so provenance is locked at
creation. Update as the corpus is built; do not retrofit.

## 1. Motivation
A market-level, longitudinal, leakage-free feature panel to (later) predict whether a *market* produces an
outlier company. No published system rates a *market* as an evolving monthly time series — that intersection
is the contribution. **This phase validates the pipeline, not predictive validity.**

## 2. Composition
- **Unit:** a panel cell `(feature, market, month)` backed by one or more bitemporal `Observation`s.
- **Market:** a team-independent market/category (not a company or founder).
- **PoC market (v1): `llm-dev-tools`** — *tooling for building LLM applications*: frameworks,
  orchestration, evaluation, serving/inference (e.g. LangChain, LlamaIndex, vLLM, DSPy, eval tooling).
  Deliberately narrow — **not** broad "AI"/"GenAI". Defining the precise term-set, package basket, and
  in/out boundary is the first build task. (The market is the *category*; venture-backed companies populate
  it; the label is whether the category produced an outlier company.)
- **Second market (deferred): `humanoid-robotics`** — added only once the llm-dev-tools pipeline runs; chosen
  because it is structurally different (hardware, slower, patent-heavy), which is what tests generalizability.
- **Cadence / span:** monthly; span TBD per data availability.

## 3. The outlier-market label  ✅ RESOLVED (with the team, 2026-06)
The target = "this market produced an outlier company." An **outlier company** (from VCBench) hits
**any** of: **≥ $500M raised**, **≥ $500M valuation**, or **≥ $500M M&A** — **not ARR** (the earlier
"$100M ARR" was a mistaken placeholder, now corrected). Newer Vela techniques also use a
**percentile-driven** target (e.g. the **90th percentile** on a chosen metric X) instead of a fixed cutoff.
`OutlierLabel` carries `metric` + `threshold_usd` (default $500M) + an optional `percentile`; evaluation
follows the VCBench standard (**F0.5**, **precision-vs-random multiple**).

The label is bitemporal: `label_knowledge_date` (when the milestone became public) must never be earlier
than the months it would inform — this prevents outcome leakage.

## 4. Collection process
Deterministic, dated-signal ingestion (v1): for each (signal, market, month T), pull raw values from
natively-dated APIs and/or the most-recent web-archive capture **strictly before end-of-T**; store the
capture/computation timestamp as `knowledge_date`. **No LLMs in v1.** Missing evidence → abstain, never
fabricate.

## 5. Sourcing rule
One dated source per value; reject any source whose earliest verifiable date is after T. v1 sources and
their known biases: arXiv (submission counts), GitHub (stars/dependency growth; stars gameable, forks
excluded), Google Trends (relative-normalized; deseasonalize), news archives (EPU/GPR recipe), corporate
filings (10-K risk share). See `signals_v1.md` for formulas.

## 6. Preprocessing / normalization
Volume-normalize then standardize (EPU recipe); correlation-based redundancy check before any dedup.

## 7. Known biases & limitations
Web archives are incomplete/lagged (~33-day median capture gap; ~20% of snapshots > 90 days stale →
flagged low-confidence). Crunchbase/funding disclosure lag. Trends is relative. **Survivorship:** the
month-T universe must include markets that later died, at their contemporaneous state. **Contamination:**
once newer models ingest a locked corpus it becomes contaminable — lock provenance at creation.

## 8. Quality control / validation
The six build-failing integrity checks (`marketbrain.validation`): **no-future-leakage** (the gate),
event-vs-knowledge skew, capture-staleness, survivorship, provenance-completeness, hindsight spot-check.
Plus a stratified **human audit** reporting the two credibility numbers: **parallel-human-index
correlation** and **false-positive rate** (GPR-style).

## 9. Provenance & immutability
Every value carries `source_url` + `content_hash` (+ archive capture timestamp). Each monthly cut is hashed
(`cut_hash`) and recorded in `data/raw/MANIFEST.tsv` (committed); data blobs are git-ignored. Append-only:
supersede by end-dating, never delete.

## 10. Maintenance
Owner: Jude (Vela). Revisions recorded by appending a new vintage (`knowledge_date`) and end-dating the
prior row. Versioned via the manifest.

## 11. Distribution / ethics
Internal Vela. Market-level — no individual PII. Reproducible from the manifest + sourcing rules.
