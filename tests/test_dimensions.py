"""Tests for the 4-dimension signal abstraction (structured-zero vs masked-absence)."""

from __future__ import annotations

from marketbrain.features.dimensions import (
    DIMENSIONS,
    channels_for,
    covered_dimensions,
    roll_up,
)


def test_dimension_channels():
    assert set(DIMENSIONS) == {"attention", "research", "adoption", "innovation"}
    assert set(channels_for("attention")) == {"wiki", "hn"}
    assert set(channels_for("adoption")) == {"gh", "pkg"}
    assert channels_for("innovation") == ("pat",)


def test_attention_is_the_universal_spine():
    # every footprint market has wiki+hn -> attention always covered
    assert "attention" in covered_dimensions(("gh", "pkg", "wiki", "hn"))      # software
    assert "attention" in covered_dimensions(("ax", "pat", "wiki", "hn"))      # deep-tech


def test_software_market_masks_innovation_not_zeros_it():
    # software: GitHub/packages, NO patents -> innovation is UNMEASURED, not 0
    dims = roll_up(("gh", "pkg", "wiki", "hn"), {"gh": 5000, "pkg": 200, "wiki": 10, "hn": 3})
    assert dims["adoption"].covered and dims["adoption"].value == (5000 + 200) / 2
    assert dims["attention"].covered
    assert not dims["innovation"].covered and dims["innovation"].value is None   # masked, not 0
    assert not dims["research"].covered                                          # no arXiv channel here


def test_deeptech_market_masks_adoption_not_zeros_it():
    # biotech: arXiv + patents, NO GitHub/packages -> adoption UNMEASURED (not "zero adoption")
    dims = roll_up(("ax", "pat", "wiki", "hn"), {"ax": 40, "pat": 120, "wiki": 8, "hn": 1})
    assert dims["research"].covered and dims["research"].value == 40
    assert dims["innovation"].covered and dims["innovation"].value == 120
    assert not dims["adoption"].covered and dims["adoption"].value is None       # masked, not 0


def test_present_but_zero_is_a_real_zero():
    # a tracked GitHub channel that happens to be 0 is a REAL zero (covered=True, value=0), not masked
    dims = roll_up(("gh", "wiki", "hn"), {"gh": 0, "wiki": 5, "hn": 2})
    assert dims["adoption"].covered and dims["adoption"].value == 0.0            # informative zero
    assert dims["adoption"].n_channels == 1


def test_missing_values_default_to_zero_for_present_channels():
    # coverage says the channel exists but no value supplied yet -> treated as 0 (present, unmeasured-as-0)
    dims = roll_up(("gh", "pkg", "wiki", "hn"))
    assert dims["adoption"].covered and dims["adoption"].value == 0.0
