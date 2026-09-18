"""Schema contracts: ALFRED aliases, immutability, dataclass<->column no-drift, month-ends."""

from __future__ import annotations

from datetime import date

import pytest

from marketbrain.schema import (
    OBSERVATION_COLUMNS,
    SUPERSEDED_OPEN,
    MarketMonth,
    observation_field_names,
)
from marketbrain.timeutil import end_of_month, require_month, to_month
from tests.conftest import make_obs


def test_alfred_aliases():
    obs = make_obs(knowledge_date=date(2024, 1, 31))
    assert obs.realtime_start == obs.knowledge_date
    assert obs.realtime_end == obs.superseded_date == SUPERSEDED_OPEN


def test_observation_is_frozen():
    obs = make_obs()
    with pytest.raises(Exception):
        obs.value = 999  # type: ignore[misc]


def test_observation_columns_match_dataclass_fields():
    # the on-disk table and the dataclasses must not drift
    assert set(OBSERVATION_COLUMNS) == observation_field_names()


def test_month_end_and_leap_year():
    assert end_of_month("2024-02") == date(2024, 2, 29)   # leap
    assert end_of_month("2023-02") == date(2023, 2, 28)
    assert end_of_month("2024-12") == date(2024, 12, 31)
    assert MarketMonth("llm-dev-tools", "2024-02").end_of() == date(2024, 2, 29)


def test_to_month_roundtrip():
    assert to_month(date(2024, 3, 9)) == "2024-03"


def test_bad_month_raises():
    with pytest.raises(ValueError):
        end_of_month("2024-13")


def test_require_month_blocks_injection():
    assert require_month("2024-03") == "2024-03"
    for bad in ["2024-03') OR '1'='1", "2024", "2024-13", "2024-3; DROP TABLE x", "abc"]:
        with pytest.raises(ValueError):
            require_month(bad)
