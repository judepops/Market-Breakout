"""Reconcile Crunchbase + Gemini into a per-company outcome and a market-level success label.

Pure decision functions (``decide_company``, ``build_market_label``) are network-free and unit-tested;
``run()`` is the orchestrator that hits BigQuery + Gemini. A company is an outlier if ANY of
raise / valuation / M&A / IPO reached the threshold; the milestone date is the **earliest** such crossing
(when it first became a >=$500M company). Grounding supplies the public citation + ``knowledge_date`` and
can gap-fill outcomes Crunchbase missed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..markets import get_market
from ..schema.features import OutlierLabel
from .crunchbase_company import CompanyFunding


@dataclass(frozen=True)
class CompanyOutcome:
    permalink: str
    name: "str | None"
    is_outlier: bool
    metric: "str | None"            # which path crossed first: valuation | raise | ma | ipo
    value_usd: "int | None"         # value of the crossing metric (or best-so-far if not an outlier)
    milestone_date: "date | None"   # when it first crossed (label_reference_date)
    knowledge_date: "date | None"   # when public (from grounding if available)
    evidence_url: "str | None"
    source: str                     # crunchbase | crunchbase+gemini | gemini | none
    verified: bool                  # grounding confirmed >= threshold with a real citation
    cb_value_usd: "int | None"      # Crunchbase best value (transparency)
    grounded_value_usd: "int | None"


def _cb_candidates(cf: CompanyFunding) -> "list[tuple[str, int, date | None]]":
    raw = [
        ("valuation", cf.valuation_usd, cf.valuation_date),
        ("raise", cf.total_raised_usd, cf.raise_cross_date),
        ("ma", cf.ma_price_usd, cf.ma_date),
        ("ipo", cf.ipo_valuation_usd, cf.ipo_date),
    ]
    return [(m, int(v), d) for (m, v, d) in raw if v]


def decide_company(
    cf: CompanyFunding, grounded=None, *, threshold_usd: int = 500_000_000
) -> CompanyOutcome:
    """Combine the Crunchbase facts and (optional) grounded result into one company outcome."""
    cands = _cb_candidates(cf)
    cb_best = max((v for _, v, _ in cands), default=None)
    qualifying = [(m, v, d) for (m, v, d) in cands if v >= threshold_usd]
    cb_outlier = bool(qualifying)
    if qualifying:  # first crossing = earliest-dated qualifying metric (undated sorts last)
        cb_metric, cb_value, cb_date = min(qualifying, key=lambda c: (c[2] is None, c[2] or date.max))
    else:
        cb_metric, cb_value, cb_date = None, cb_best, None

    g = grounded
    g_outlier = bool(g and g.verified)

    if cb_outlier and g_outlier:
        source, verified = "crunchbase+gemini", True
        metric, value, milestone = cb_metric, cb_value, cb_date
        knowledge, url = (g.public_date or g.milestone_date or cb_date), g.evidence_url
    elif cb_outlier:
        source, verified = "crunchbase", False  # CB says yes, grounding did not confirm/cite
        metric, value, milestone = cb_metric, cb_value, cb_date
        knowledge, url = cb_date, (g.evidence_url if g else None)
    elif g_outlier:  # gap-fill: CB missed it, grounding found a cited >= threshold outcome
        source, verified = "gemini", True
        metric, value, milestone = g.metric, g.value_usd, g.milestone_date
        knowledge, url = (g.public_date or g.milestone_date), g.evidence_url
    else:
        source, verified = ("crunchbase" if cands else "none"), False
        metric, value, milestone, knowledge = None, cb_best, None, None
        url = g.evidence_url if g else None

    # an outcome can't be "known public" before it occurred: knowledge_date >= milestone_date
    knowledge = max((d for d in (milestone, knowledge) if d is not None), default=None)

    return CompanyOutcome(
        permalink=cf.permalink, name=cf.name, is_outlier=cb_outlier or g_outlier,
        metric=metric, value_usd=value, milestone_date=milestone, knowledge_date=knowledge,
        evidence_url=url, source=source, verified=verified,
        cb_value_usd=cb_best, grounded_value_usd=(g.value_usd if g else None),
    )


def build_market_label(
    outcomes: "list[CompanyOutcome]", market: str, *,
    threshold_usd: int = 500_000_000, horizon_years: float = 5.0,
) -> OutlierLabel:
    """Market is an outlier if any company crossed; reference date = earliest company crossing."""
    winners = [o for o in outcomes if o.is_outlier]
    dated = [o for o in winners if o.milestone_date]
    first = min(dated, key=lambda o: o.milestone_date) if dated else (winners[0] if winners else None)
    return OutlierLabel(
        market=market, is_outlier=bool(winners),
        threshold_usd=float(threshold_usd), horizon_years=horizon_years,
        label_reference_date=first.milestone_date if first else None,
        label_knowledge_date=first.knowledge_date if first else None,
        evidence_url=first.evidence_url if first else None,
    )


def run(
    market: str = "llm-dev-tools", permalinks: "tuple[str, ...] | None" = None, *,
    threshold_usd: int = 500_000_000, ground: bool = True, project: "str | None" = None,
    model: str = "gemini-2.5-flash", verbose: bool = True,
) -> "tuple[OutlierLabel, list[CompanyOutcome]]":
    """End-to-end: Crunchbase pull -> (optional) Gemini grounding -> outcomes -> market label."""
    from .crunchbase_company import pull

    mod = get_market(market)
    permalinks = permalinks or mod.permalinks()
    names = dict(zip(mod.permalinks(), mod.names()))
    cb = pull(permalinks, threshold_usd=threshold_usd, project=project, verbose=verbose)

    client = ground_fn = None
    if ground:
        from .gemini_grounding import ground_company, make_client
        client, ground_fn = make_client(), ground_company

    outcomes = []
    for p in permalinks:
        cf = cb.get(p) or CompanyFunding(p, names.get(p), None, None, None, None, None, None, None, None)
        grounded = None
        if ground:
            grounded = ground_fn(cf.name or names.get(p) or p, p,
                                 threshold_usd=threshold_usd, client=client, model=model)
        outcome = decide_company(cf, grounded, threshold_usd=threshold_usd)
        outcomes.append(outcome)
        if verbose:
            tag = "OUTLIER" if outcome.is_outlier else "—"
            val = f"${outcome.value_usd/1e9:.2f}B" if outcome.value_usd else "n/a"
            print(f"  {p:24s} {tag:8s} {str(outcome.metric or ''):10s} {val:9s} "
                  f"[{outcome.source}{'' if outcome.verified else ' UNVERIFIED'}]")

    label = build_market_label(outcomes, market, threshold_usd=threshold_usd)
    return label, outcomes
