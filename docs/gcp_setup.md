# GCP / BigQuery setup — for the PyPI adoption signal

The PyPI download signal (`marketbrain/signals/pypi.py`) reads the BigQuery public dataset
`bigquery-public-data.pypi.file_downloads`. That needs Google Cloud auth + a billing/quota project.
This is the **Python** adoption half (the one that matters for this Python-heavy market); npm covers
only JS. Project used: **`<your-gcp-project>`**.

## One-time cluster setup

```bash
# 1. install the gcloud SDK (non-interactive)
curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir="$HOME"

# 2. Application Default Credentials (headless: prints a URL, paste the code back)
~/google-cloud-sdk/bin/gcloud auth application-default login --no-launch-browser
#    -> on the consent screen TICK EVERY BOX (esp. "Cloud Platform"); a partial grant
#       fails with "Scope has changed ... to userinfo.email".

# 3. attach a quota/billing project (BigQuery bills a project even for public data)
~/google-cloud-sdk/bin/gcloud auth application-default set-quota-project <your-gcp-project>

# 4. python client (already in the marketbrain env)
pip install google-cloud-bigquery db-dtypes
```

ADC lands at `~/.config/gcloud/application_default_credentials.json` — **outside the repo, never
commit it** (secrets rule). `bigquery.Client(project="<your-gcp-project>")` then picks it up.

## Cost discipline (BigQuery is metered; free tier = 1 TB scanned/month)
The table is **partitioned by `timestamp` (DAY)** and **clustered by `project`**, so the collector:
- filters with a **partition-pruning** range `timestamp >= … AND timestamp < …` (never `DATE(timestamp)`,
  which scans every partition) plus a `project IN (...)` filter (clustering keeps the scan small);
- **dry-runs** first to log the upper-bound bytes;
- sets **`maximum_bytes_billed`** as a runaway guard (a too-big query fails at **zero charge**);
- **caches** the monthly result to `data/raw/pypi_monthly_<start>_<end>.parquet` (git-ignored) so the
  query runs **at most once** per (start, end, basket). Delete the cache to force a refresh.

Reference cost: the 7-package full backfill `2022-06..2025-12` billed **~315 GB** (one-time, free-tier).

## Run it
```bash
python -m experiments.collect_signal pypi --start 2022-06 --end 2025-12 --root data/raw
python -m experiments.plot_signal   pypi --root data/raw --out artifacts/figures
```
