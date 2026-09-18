"""Month/period helpers. The leakage cutoff depends on these being exactly right
(month-ends, leap years), so they live in one place and are unit-tested directly.
"""

from __future__ import annotations

from datetime import date, timedelta

MONTH_FMT = "%Y-%m"


def end_of_month(month: str) -> date:
    """Last calendar day of ``month`` ("YYYY-MM"). This is the leakage cutoff for
    the month-T panel column: every contributing ``knowledge_date`` must be <= this.
    """
    year, mon = _parse_month(month)
    first_of_next = date(year + 1, 1, 1) if mon == 12 else date(year, mon + 1, 1)
    return first_of_next - timedelta(days=1)


def to_month(d: date) -> str:
    """Canonical "YYYY-MM" string for a date."""
    return f"{d.year:04d}-{d.month:02d}"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of "YYYY-MM" strings from ``start`` to ``end``."""
    y, m = _parse_month(start)
    ey, em = _parse_month(end)
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def require_month(month: str) -> str:
    """Validate ``month`` is exactly 'YYYY-MM' (raise ValueError otherwise) and return it.

    Guard for values that get interpolated into a query string — ``_parse_month`` parses the
    components as ints and range-checks the month, so no SQL metacharacters can survive.
    """
    _parse_month(month)
    return month


def _parse_month(month: str) -> tuple[int, int]:
    try:
        year_s, mon_s = month.split("-")
        year, mon = int(year_s), int(mon_s)
    except (ValueError, AttributeError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"month must be 'YYYY-MM', got {month!r}") from exc
    if not 1 <= mon <= 12:
        raise ValueError(f"month out of range in {month!r}")
    return year, mon
