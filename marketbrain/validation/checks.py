"""Build-failing integrity checks (Stage 0).

One pure function per check from ``Docs/corpus_methodology.md`` ("Build-failing
integrity checks"). Each takes a DataFrame and returns a list of ``IntegrityFailure``
— it never raises. ``runner.run_all_checks`` aggregates and decides whether to fail
the build. ``check_no_future_leakage`` is THE check; everything else supports it.

Panel-level checks operate on the normalized panel (``PANEL_COLUMNS``); observation-
level checks operate on the raw bitemporal log (``OBSERVATION_COLUMNS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..timeutil import end_of_month, to_month

ERROR = "error"
WARNING = "warning"

# Deterministic lexicon for the v1 hindsight scan (upgraded to an LLM judge later).
_HINDSIGHT_PHRASES = (
    "became the leader",
    "market leader",
    "the winner",
    "went on to",
    "would later",
    "would become",
    "eventually became",
    "turned out to be",
    "in hindsight",
    "as we now know",
    "ultimately won",
    "dominated the market",
)


@dataclass(frozen=True)
class IntegrityFailure:
    check: str
    severity: str  # ERROR (fails the build) | WARNING
    detail: str
    observation_id: Optional[str] = None
    market: Optional[str] = None
    month: Optional[str] = None


class IntegrityError(AssertionError):
    """Raised by ``run_all_checks`` when any ERROR-severity failure is present."""

    def __init__(self, failures: list[IntegrityFailure]):
        self.failures = failures
        errs = [f for f in failures if f.severity == ERROR]
        super().__init__(f"{len(errs)} integrity error(s) — build must fail")


# --------------------------------------------------------------------------- #
# THE critical check
# --------------------------------------------------------------------------- #
def check_no_future_leakage(panel: pd.DataFrame) -> list[IntegrityFailure]:
    """Every cell in the month-T column must have ``knowledge_date_max <= end_of(T)``.

    A planted future-dated cell MUST produce an error here — this is the single
    most important guarantee in the project. A missing ``knowledge_date_max`` is
    also an error (a cell whose leakage-safety can't be established).
    """
    out: list[IntegrityFailure] = []
    if panel.empty:
        return out
    cutoffs = panel["month"].map(lambda m: pd.Timestamp(end_of_month(m)))
    kd = pd.to_datetime(panel["knowledge_date_max"])

    for idx in panel.index[kd.isna()]:
        row = panel.loc[idx]
        out.append(IntegrityFailure(
            "no_future_leakage", ERROR,
            "cell has no knowledge_date_max; leakage-safety cannot be established",
            market=row.get("market"), month=row.get("month"),
        ))
    leaked = kd > cutoffs
    for idx in panel.index[leaked.fillna(False)]:
        row = panel.loc[idx]
        out.append(IntegrityFailure(
            "no_future_leakage", ERROR,
            f"knowledge_date_max {kd[idx].date()} is after end_of({row['month']}) "
            f"= {cutoffs[idx].date()} — future leakage",
            market=row.get("market"), month=row.get("month"),
        ))
    return out


# --------------------------------------------------------------------------- #
# observation-level checks
# --------------------------------------------------------------------------- #
def check_event_vs_knowledge_skew(
    observations: pd.DataFrame, max_backreport_days: int = 3650
) -> list[IntegrityFailure]:
    """Error if ``knowledge_date < reference_date`` (knowing before it happened).
    Warning if the back-reporting gap exceeds ``max_backreport_days``.
    """
    out: list[IntegrityFailure] = []
    if observations.empty:
        return out
    ref = pd.to_datetime(observations["reference_date"])
    kno = pd.to_datetime(observations["knowledge_date"])
    gap_days = (kno - ref).dt.days

    for idx in observations.index[(kno < ref).fillna(False)]:
        r = observations.loc[idx]
        out.append(IntegrityFailure(
            "event_vs_knowledge_skew", ERROR,
            f"knowledge_date {kno[idx].date()} precedes reference_date {ref[idx].date()}",
            observation_id=r.get("observation_id"), market=r.get("market"),
        ))
    for idx in observations.index[(gap_days > max_backreport_days).fillna(False)]:
        r = observations.loc[idx]
        out.append(IntegrityFailure(
            "event_vs_knowledge_skew", WARNING,
            f"large back-reporting gap: {int(gap_days[idx])} days",
            observation_id=r.get("observation_id"), market=r.get("market"),
        ))
    return out


def check_capture_staleness(
    observations: pd.DataFrame, stale_days: int = 90
) -> list[IntegrityFailure]:
    """Warning where an archive capture is more than ``stale_days`` before the end
    of the period it grounds (Wayback's lag problem). Low-confidence, not fatal.
    """
    out: list[IntegrityFailure] = []
    if observations.empty or "archive_capture_timestamp" not in observations:
        return out
    cap = pd.to_datetime(observations["archive_capture_timestamp"])
    ref = pd.to_datetime(observations["reference_date"])
    has_cap = cap.notna() & ref.notna()
    for idx in observations.index[has_cap]:
        target_end = pd.Timestamp(end_of_month(to_month(ref[idx].date())))
        gap = (target_end - cap[idx]).days
        if gap > stale_days:
            r = observations.loc[idx]
            out.append(IntegrityFailure(
                "capture_staleness", WARNING,
                f"capture {cap[idx].date()} is {gap} days before end of target period",
                observation_id=r.get("observation_id"), market=r.get("market"),
            ))
    return out


def check_provenance_completeness(observations: pd.DataFrame) -> list[IntegrityFailure]:
    """Error for any value-bearing observation missing ``source_url`` or
    ``content_hash``. Abstentions (qc_status == 'abstained') are exempt.
    """
    out: list[IntegrityFailure] = []
    if observations.empty:
        return out
    for idx, r in observations.iterrows():
        if str(r.get("qc_status")) == "abstained":
            continue
        if _blank(r.get("source_url")) or _blank(r.get("content_hash")):
            out.append(IntegrityFailure(
                "provenance_completeness", ERROR,
                "value-bearing observation missing source_url and/or content_hash",
                observation_id=r.get("observation_id"), market=r.get("market"),
            ))
    return out


def check_hindsight_spotcheck(observations: pd.DataFrame) -> list[IntegrityFailure]:
    """Warning for any ``claim_text`` asserting an outcome not yet public at T.
    Deterministic lexicon scan in v1; an LLM judge replaces it in Stage 4.
    """
    out: list[IntegrityFailure] = []
    if observations.empty or "claim_text" not in observations:
        return out
    for idx, r in observations.iterrows():
        text = r.get("claim_text")
        if not isinstance(text, str):
            continue
        low = text.lower()
        hit = next((p for p in _HINDSIGHT_PHRASES if p in low), None)
        if hit:
            out.append(IntegrityFailure(
                "hindsight_spotcheck", WARNING,
                f"claim_text contains hindsight phrase {hit!r}",
                observation_id=r.get("observation_id"), market=r.get("market"),
            ))
    return out


def check_coverage_completeness(
    observations: pd.DataFrame,
    *,
    neighbour_window: int = 3,
    min_fraction: float = 0.25,
    min_median: float = 100.0,
) -> list[IntegrityFailure]:
    """Warning for any month whose value is materially below its **recent trailing months**
    in the same (feature, market) series — a sign of **incomplete source coverage** (e.g. a
    partial-month aggregate, like the Wikipedia Dec-2025 dip that was ~3% of the months before it).

    Compares to the *preceding* months (not a centred window) on purpose: that catches a sharp
    drop from the established level — including an end-of-series partial month (the actual bug) —
    without false-flagging a genuine *ramp-up* (where the preceding months are legitimately low,
    e.g. the pre-ChatGPT period). Heuristic → a warning, not an error. Sparse / low-volume series
    (median < ``min_median``) are skipped, since for a sparse count signal (funding Form-D counts,
    story counts) a low month is normal, not incomplete; the check targets high-volume aggregates
    (pageviews, downloads, submission counts).
    """
    out: list[IntegrityFailure] = []
    if observations.empty:
        return out
    df = observations.copy()
    df["reference_date"] = pd.to_datetime(df["reference_date"])

    for (feature, market), grp in df.groupby(["feature", "market"]):
        grp = grp.sort_values("reference_date")
        values = grp["value"].astype(float).tolist()
        months = [d.strftime("%Y-%m") for d in grp["reference_date"]]
        ids = grp["observation_id"].tolist()
        if not values:
            continue
        if sorted(values)[len(values) // 2] < min_median:  # sparse/low-volume series — skip
            continue
        for i in range(len(values)):
            trailing = values[max(0, i - neighbour_window):i]  # the months immediately before i
            if len(trailing) < 2:
                continue
            tr_median = sorted(trailing)[len(trailing) // 2]
            if tr_median > 0 and values[i] < min_fraction * tr_median:
                out.append(IntegrityFailure(
                    "coverage_completeness", WARNING,
                    f"{feature} value {values[i]:.0f} at {months[i]} is < {min_fraction:.0%} of the "
                    f"trailing {len(trailing)}-month median {tr_median:.0f} — possible incomplete coverage",
                    observation_id=ids[i], market=market, month=months[i],
                ))
    return out


# --------------------------------------------------------------------------- #
# panel vs universe
# --------------------------------------------------------------------------- #
def check_survivorship(panel: pd.DataFrame, universe: pd.DataFrame) -> list[IntegrityFailure]:
    """The month-T universe must be the markets active AT T (including ones that
    later died). ``universe`` has columns (market, active_from, active_to) as
    "YYYY-MM" strings (active_to may be None/NaT for still-active).

    Error if a panel cell exists for a market not active at its month, or if a
    market active at month T has no cell at all (panel built from a survivor list).
    """
    out: list[IntegrityFailure] = []
    if panel.empty:
        return out
    uni = universe.copy()
    present = set(zip(panel["market"], panel["month"]))
    months = sorted(panel["month"].unique())

    for _, u in uni.iterrows():
        mkt = u["market"]
        a_from = str(u.get("active_from")) if pd.notna(u.get("active_from")) else None
        a_to = u.get("active_to")
        a_to = str(a_to) if (a_to is not None and pd.notna(a_to)) else None
        for m in months:
            active = (a_from is None or m >= a_from) and (a_to is None or m <= a_to)
            if active and (mkt, m) not in present:
                out.append(IntegrityFailure(
                    "survivorship", ERROR,
                    f"market active at {m} but absent from the panel "
                    "(panel may be built from a present-day survivor list)",
                    market=mkt, month=m,
                ))

    active_lookup = {
        (u["market"]): (
            str(u.get("active_from")) if pd.notna(u.get("active_from")) else None,
            str(u.get("active_to")) if (u.get("active_to") is not None
                                        and pd.notna(u.get("active_to"))) else None,
        )
        for _, u in uni.iterrows()
    }
    for mkt, m in present:
        if mkt not in active_lookup:
            out.append(IntegrityFailure(
                "survivorship", ERROR,
                "panel cell for a market absent from the contemporaneous universe",
                market=mkt, month=m,
            ))
            continue
        a_from, a_to = active_lookup[mkt]
        if (a_from is not None and m < a_from) or (a_to is not None and m > a_to):
            out.append(IntegrityFailure(
                "survivorship", ERROR,
                f"panel cell at {m} for a market not active at that month",
                market=mkt, month=m,
            ))
    return out


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "" \
        or str(v) == "<NA>"
