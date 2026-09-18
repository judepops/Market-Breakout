"""Stage B-B, step 1: pull Crunchbase company descriptions for the semantic theme->company bridge.

    GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp/production-498516-defd8abe6f50.json \
        python -m experiments.stageB_pull_descriptions

Descriptions are NOT cached anywhere (every company cache is permalink+numbers), so the bridge needs a live
pull. We pull short_description + description + category tags for the 34,417 companies already in our roster
(company_meta.parquet), which is exactly the set that has funding (company_rounds) and outcomes (universes).
Cost-guarded (maximum_bytes_billed), chunked, cached to data/raw/labels/company_descriptions.parquet.
"""
from __future__ import annotations
import pathlib

DATASET = "production-498516.Crunchbase_Dataset"
MAX_BYTES = 8 * 10**9
OUT = pathlib.Path("data/raw/labels/company_descriptions.parquet")


def main():
    import pandas as pd
    from google.cloud import bigquery

    roster = pd.read_parquet("data/raw/labels/company_meta.parquet")
    permalinks = sorted(roster["permalink"].dropna().astype(str).unique())
    print(f"[descr] pulling descriptions for {len(permalinks):,} companies", flush=True)

    client = bigquery.Client()
    sql = f"""
    SELECT identifier_permalink AS permalink,
           identifier_value      AS name,
           short_description,
           description,
           categories_value      AS categories,
           category_groups_value AS category_groups
    FROM `{DATASET}.organizations`
    WHERE identifier_permalink IN UNNEST(@permalinks)
    """
    frames = []
    CH = 8000
    for i in range(0, len(permalinks), CH):
        chunk = permalinks[i:i + CH]
        cfg = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("permalinks", "STRING", chunk)],
            maximum_bytes_billed=MAX_BYTES)
        df = client.query(sql, job_config=cfg).to_dataframe()
        frames.append(df)
        print(f"[descr]   chunk {i // CH + 1}/{-(-len(permalinks) // CH)}: +{len(df)} rows "
              f"(cum {sum(len(f) for f in frames):,})", flush=True)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("permalink")
    # combined text field for embedding (name + short_desc + categories; fall back through fields)
    def _txt(r):
        parts = [str(r["name"] or "")]
        sd = r["short_description"] or r["description"] or ""
        parts.append(str(sd))
        cat = r["categories"] or r["category_groups"] or ""
        if cat:
            parts.append("Categories: " + str(cat))
        return " . ".join(p for p in parts if p).strip()
    out["bridge_text"] = out.apply(_txt, axis=1)
    n_desc = (out["short_description"].fillna(out["description"]).fillna("").str.len() > 0).sum()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"[descr] wrote {OUT}: {len(out):,} companies, {n_desc:,} with a real description", flush=True)


if __name__ == "__main__":
    main()
