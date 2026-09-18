"""Calibrate each market's full-universe tag spec — the saturation fix's definitional step.

    python -m experiments.calibrate_universe

For each candidate tag spec we report: universe size (>= $5M fundable), how many of the curated roster
winners it captures (it should contain the known winners), the outlier count + RATE, and a few example
NEGATIVES (fundable companies that did NOT cross $500M — the variation the curated rosters lacked). Pick the
spec that captures the winners while admitting real negatives, then lock it into the market module.
"""

from __future__ import annotations

from marketbrain.markets import data_infra, llm_dev_tools
from marketbrain.markets.universe import fetch_universe, outlier_rate

# Candidate specs per market: (label, tag_groups, founded_from). Empirical — resolved by the numbers below.
CANDIDATES = {
    "llm-dev-tools": [
        ("Generative AI only", [["Generative AI"]], None),
        ("AI Infrastructure only", [["AI Infrastructure"]], None),
        ("AI AND (infra|genai|devtools)",
         [["Artificial Intelligence (AI)"], ["AI Infrastructure", "Generative AI", "Developer Tools"]], None),
        ("AI AND (infra|genai|devtools), founded>=2018",
         [["Artificial Intelligence (AI)"], ["AI Infrastructure", "Generative AI", "Developer Tools"]], 2018),
    ],
    "data-infra": [
        ("Database|Analytics|Data Storage", [["Database", "Analytics", "Data Storage"]], None),
        ("Database|Data Storage|Data Mgmt", [["Database", "Data Storage", "Data Management"]], None),
        ("(DB|Analytics) AND Software", [["Database", "Analytics", "Data Storage"], ["Software"]], None),
    ],
}
ROSTERS = {"llm-dev-tools": llm_dev_tools, "data-infra": data_infra}


def main() -> int:
    for market, cands in CANDIDATES.items():
        winners = set(ROSTERS[market].permalinks())
        print(f"\n{'='*92}\n{market}: curated roster = {len(winners)} companies\n{'='*92}")
        print(f"{'candidate spec':46s} {'univ':>6s} {'capt':>6s} {'>=500M':>7s} {'rate':>7s}")
        for label, groups, founded in cands:
            comps, billed = fetch_universe(groups, founded_from=founded)
            perms = {c.permalink for c in comps}
            captured = len(winners & perms)
            rate, o, n = outlier_rate(comps)
            print(f"{label:46s} {n:>6,} {captured:>3d}/{len(winners):<2d} {o:>7,} {rate:>6.1%}"
                  f"   [{billed/1e6:.0f} MB]")
            missed = winners - perms
            if missed and captured < len(winners):
                print(f"{'':46s} missed winners: {', '.join(sorted(missed))[:120]}")
    print("\nPick the spec that best captures winners AND yields real negatives (rate well under 100%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
