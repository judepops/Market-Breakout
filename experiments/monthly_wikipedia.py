"""Broad-attention test: monthly Wikipedia pageviews per market (vs HN's developer niche).

    python -m experiments.monthly_wikipedia

Wikipedia pageviews are broad attention for any topic (fintech/robotics register here, unlike HN). An LLM
resolves each market to its canonical en.wikipedia article; the Wikimedia pageviews API gives monthly views
(2015-07 onward). Macro-normalised by the within-corpus total. Resumable article cache.
"""
from __future__ import annotations
import json
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/user/{a}/monthly/2015070100/2026060100")
OUT = Path("data/raw/timeseries")
ARTS = Path("data/raw/labels/wikipedia_articles.json")


def _pageviews(article: str):
    url = API.format(a=urllib.parse.quote(article.replace(" ", "_"), safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research; jude)"})
    for attempt in range(4):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=30).read())
            return [(int(it["timestamp"][:4]), int(it["timestamp"][4:6]), int(it["views"]))
                    for it in data.get("items", [])]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * 2 ** attempt)
        except Exception:  # noqa: BLE001
            time.sleep(2 * 2 ** attempt)
    return None


def main():
    import pandas as pd
    from marketbrain.markets.partition import _KEY_TO_TAG
    from marketbrain.labels.gemini_grounding import make_client
    OUT.mkdir(parents=True, exist_ok=True)

    markets = list(pd.read_parquet("data/raw/labels/partition_features.parquet")["market"])
    arts = json.loads(ARTS.read_text()) if ARTS.exists() else {}
    client = make_client()
    for mk in markets:                                          # LLM -> canonical article title
        if mk not in arts:
            name = _KEY_TO_TAG.get(mk, mk.replace("-", " "))
            prompt = (f'What is the exact English Wikipedia article title for the "{name}" technology topic? '
                      f'Return ONLY the article title, nothing else.')
            try:
                arts[mk] = client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt, config={"temperature": 0}).text.strip().strip('"')
            except Exception:  # noqa: BLE001
                arts[mk] = name
            ARTS.write_text(json.dumps(arts))

    rows, miss = [], []
    for mk in markets:
        pv = _pageviews(arts[mk])
        time.sleep(0.2)
        if pv:
            for y, mo, v in pv:
                rows.append({"market": mk, "year": y, "month": mo, "views": v})
        else:
            miss.append(f"{mk}({arts[mk]})")
    df = pd.DataFrame(rows)
    tot = df.groupby(["year", "month"])["views"].transform("sum")
    df["share"] = df["views"] / tot
    df.to_parquet(OUT / "monthly_wikipedia.parquet", index=False)
    print(f"[wikipedia] {df['market'].nunique()} markets with pageviews; {len(miss)} missing: {miss[:8]}")
    print(f"[wikipedia] wrote {len(df)} rows -> {OUT/'monthly_wikipedia.parquet'}")


if __name__ == "__main__":
    main()
