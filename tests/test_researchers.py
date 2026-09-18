import numpy as np
import pandas as pd

from marketbrain.eval.researchers import theme_key_papers, theme_researchers


def _author_index():
    # Ada: 36 prior papers across 2019-2021 (all before 2023). Bob: 2. Carol: none.
    ada = np.array(sorted(f"{y}-{m:02d}" for y in (2019, 2020, 2021) for m in range(1, 13)))
    return {"Ada": ada, "Bob": np.array(["2018-01", "2018-06"])}


def _recs():
    return pd.DataFrame({
        "record_id": ["http://arxiv.org/abs/2301.001", "http://arxiv.org/abs/2305.002",
                      "http://arxiv.org/abs/2312.003"],
        "month": ["2023-01", "2023-05", "2023-12"],
        "title": ["Founding paper", "Second paper", "Late paper"],
        "authors": [["Ada", "Carol"], ["Ada"], ["Bob"]],
    })


def test_theme_researchers_ranks_by_track_record_and_involvement():
    res = theme_researchers(_recs(), t0="2023-01", author_index=_author_index())
    names = [r["name"] for r in res]
    assert names[0] == "Ada"  # 2 theme papers + big prior + early
    ada = res[0]
    assert ada["theme_papers"] == 2 and ada["prior_papers"] == 36 and ada["early"] is True
    carol = next(r for r in res if r["name"] == "Carol")
    assert carol["prior_papers"] == 0 and carol["early"] is True
    bob = next(r for r in res if r["name"] == "Bob")
    assert bob["early"] is False  # only appears at 2023-12, past the 6-month early window


def test_theme_key_papers_ranks_prestige_first_low_last_with_links():
    papers = theme_key_papers(_recs(), _author_index(), t0="2023-01")
    assert "Ada" in papers[0]["authors"]                 # a high-prestige Ada paper leads
    assert papers[-1]["title"] == "Late paper"           # Bob (low prestige) last
    assert papers[0]["url"].startswith("http://arxiv.org/abs/")


def test_key_papers_when_record_id_is_the_index():
    # the real caller passes records indexed BY record_id, not as a column
    recs = _recs().set_index("record_id")
    papers = theme_key_papers(recs, _author_index(), t0="2023-01")
    assert papers[0]["url"].startswith("http://arxiv.org/abs/")
    assert len(papers) == 3


def test_empty_recs():
    empty = pd.DataFrame(columns=["record_id", "month", "title", "authors"])
    assert theme_researchers(empty, "2023-01", {}) == []
    assert theme_key_papers(empty, {}, "2023-01") == []
