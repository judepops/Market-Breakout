"""The ``data-infra`` market — databases / data-infrastructure (the feasibility-spike second market).

A deliberately *older, OSS-heavy* market (most companies founded ~2010-2016) so it has the **feature ->
outcome time gap** that ``llm-dev-tools`` structurally lacked: emergence/adoption years before the >=$500M
outcomes (2018-2022). It also validates the **patent** signal that read ~0 for LLM dev-tools — databases
patent meaningfully.

Same interface as ``llm_dev_tools`` (``permalinks`` / ``names`` / ``assignee_names`` / ``github_repos``) so
every signal + the label run through it unchanged. Roster auto-seeded from the Crunchbase ``Database`` /
``Analytics`` categories (top by valuation) then **hand-curated** to drop the category noise (London Stock
Exchange, insurers, healthcare) and add the giants tagged elsewhere (Snowflake/Databricks/Confluent/dbt).
Permalinks verified 2026-06 against ``production-498516.Crunchbase_Dataset.organizations``.
"""

from __future__ import annotations

from .llm_dev_tools import Company

ROSTER: tuple[Company, ...] = (
    Company("Databricks", "databricks", "data-platform", ()),
    Company("Snowflake", "snowflake-computing", "data-warehouse", ()),
    Company("Confluent", "confluent", "streaming", ()),
    Company("dbt Labs", "dbt-labs", "elt", ()),
    Company("Fivetran", "fivetran", "elt", ()),
    Company("Dremio", "dremio-corporation", "query-engine", ()),
    Company("DataStax", "datastax", "database", ()),
    Company("Elastic", "elasticsearch", "search", ()),
    Company("Cockroach Labs", "cockroach-labs", "database", ()),
    Company("MongoDB", "mongodb-inc", "database", ()),
    Company("Redis", "redis", "database", ()),
    Company("Neo4j", "neo-technology", "graph-db", ()),
    Company("ClickHouse", "clickhouse", "olap", ()),
    Company("Starburst", "starburst", "query-engine", ()),
    Company("Aiven", "aiven", "managed-db", ()),
    Company("PingCAP", "pingcap", "database", ()),
    Company("SingleStore", "singlestore", "database", ()),
    Company("Yugabyte", "yugabyte", "database", ()),
    Company("Timescale", "timescaledb", "database", ()),
    Company("Materialize", "materialize-38fc", "streaming-db", ()),
    Company("Redpanda", "redpanda", "streaming", ()),
)

# Anchor OSS repos for the GitHub stars signal (the major open-source projects of the roster; closed-source
# companies like Snowflake/Fivetran/Aiven contribute little GitHub signal).
GITHUB_REPOS: tuple[str, ...] = (
    "elastic/elasticsearch", "mongodb/mongo", "redis/redis", "cockroachdb/cockroach",
    "ClickHouse/ClickHouse", "neo4j/neo4j", "dbt-labs/dbt-core", "pingcap/tidb",
    "yugabyte/yugabyte-db", "timescale/timescaledb", "MaterializeInc/materialize",
    "redpanda-data/redpanda", "trinodb/trino", "dremio/dremio-oss", "delta-io/delta", "mlflow/mlflow",
)


def permalinks() -> tuple[str, ...]:
    return tuple(c.cb_permalink for c in ROSTER)


def names() -> tuple[str, ...]:
    return tuple(c.name for c in ROSTER)


def assignee_names() -> tuple[str, ...]:
    """Patent-assignee match names (core company names) — databases patent meaningfully, unlike LLM tools."""
    return tuple(c.name.split(" (")[0] for c in ROSTER)


def github_repos() -> tuple[str, ...]:
    return GITHUB_REPOS


# Full-universe spec (the saturation fix) — every fundable (>= $5M) database / analytics / data-storage
# company, winners AND non-winners, for the outlier-RATE label. Calibrated 2026-06 (experiments/
# calibrate_universe.py): 4,602 companies, captures 19/21 curated winners, **10.4% outlier rate** (vs the
# curated roster's saturated 100%). The curated ROSTER above stays the *signal* anchor (GitHub repos, patent
# assignees); the universe is the *outcome/label* scope.
UNIVERSE_TAG_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Database", "Analytics", "Data Storage"),
)
UNIVERSE_FOUNDED_FROM: "int | None" = None


def universe(project: "str | None" = None):
    """The full fundable (>= $5M) universe for the outlier-rate label (winners AND negatives)."""
    from .universe import fetch_universe
    comps, _ = fetch_universe([list(g) for g in UNIVERSE_TAG_GROUPS],
                              founded_from=UNIVERSE_FOUNDED_FROM, project=project)
    return comps
