from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from aix_page.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data/processed/aix_page.duckdb"
FACT_GLOB = (ROOT / "data/staged/kdd/fact_sponsored_search/*.parquet").as_posix()
LOOKUP_ROOT = ROOT / "data/staged/kdd/lookups"
QUALITY = ROOT / "artifacts/qualification/kdd_data_quality.json"
LINEAGE = ROOT / "data/manifests/kdd_duckdb_lineage.json"


def scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def main() -> None:
    free_before = shutil.disk_usage(ROOT).free
    if free_before - 16 * 1024**3 < 50 * 1024**3:
        raise RuntimeError("DuckDB mart projection violates 50 GiB floor")
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    temp = ROOT / "data/processed/duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection = duckdb.connect(str(DATABASE))
    connection.execute("SET threads=8")
    connection.execute(f"SET temp_directory='{temp.as_posix()}'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW fact_sponsored_search AS
        SELECT Click::BIGINT AS click, Impression::BIGINT AS impression,
               DisplayURL::UBIGINT AS display_url_id, AdID::UBIGINT AS ad_id,
               AdvertiserID::UBIGINT AS advertiser_id, Depth::INTEGER AS depth,
               Position::INTEGER AS position, QueryID::UBIGINT AS query_id,
               KeywordID::UBIGINT AS keyword_id, TitleID::UBIGINT AS title_id,
               DescriptionID::UBIGINT AS description_id, UserID::UBIGINT AS user_id,
               hash(UserID, AdID, QueryID, Position, Depth) AS setting_hash,
               CASE
                 WHEN hash(UserID, AdID, QueryID, Position, Depth) % 100 < 50 THEN 'model_train'
                 WHEN hash(UserID, AdID, QueryID, Position, Depth) % 100 < 60 THEN 'model_validation'
                 WHEN hash(UserID, AdID, QueryID, Position, Depth) % 100 < 70 THEN 'model_calibration'
                 WHEN hash(UserID, AdID, QueryID, Position, Depth) % 100 < 80 THEN 'model_locked_test'
                 WHEN hash(UserID, AdID, QueryID, Position, Depth) % 100 < 95 THEN 'experiment_context_pool'
                 ELSE 'challenge_pool'
               END AS data_role
        FROM read_parquet('{FACT_GLOB}', union_by_name=true)
        """
    )
    connection.execute(
        "CREATE OR REPLACE TABLE dim_ad AS SELECT ad_id, any_value(advertiser_id) advertiser_id, "
        "any_value(title_id) title_id, any_value(description_id) description_id, count(*) support "
        "FROM fact_sponsored_search GROUP BY ad_id"
    )
    connection.execute(
        "CREATE OR REPLACE TABLE dim_advertiser AS SELECT advertiser_id, count(*) support "
        "FROM fact_sponsored_search GROUP BY advertiser_id"
    )
    connection.execute(
        "CREATE OR REPLACE TABLE dim_display_url AS SELECT display_url_id, count(*) support "
        "FROM fact_sponsored_search GROUP BY display_url_id"
    )
    connection.execute(
        "CREATE OR REPLACE TABLE dim_query AS SELECT query_id, any_value(keyword_id) keyword_id, "
        "count(*) support FROM fact_sponsored_search GROUP BY query_id"
    )
    connection.execute(
        "CREATE OR REPLACE TABLE dim_user AS SELECT user_id, user_id=0 AS is_anonymous, "
        "count(*) support FROM fact_sponsored_search GROUP BY user_id"
    )
    for table, folder, columns in [
        ("lookup_query_tokens", "query", "id AS query_id, tokens"),
        ("lookup_keyword_tokens", "keyword", "id AS keyword_id, tokens"),
        ("lookup_title_tokens", "title", "id AS title_id, tokens"),
        ("lookup_description_tokens", "description", "id AS description_id, tokens"),
        ("lookup_user_profile", "user_profile", "id AS user_id, gender, age"),
    ]:
        glob = (LOOKUP_ROOT / folder / "*.parquet").as_posix()
        connection.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT {columns} FROM read_parquet('{glob}')"
        )
    connection.execute(
        """
        CREATE OR REPLACE VIEW mart_ctr_features AS
        SELECT *, position::DOUBLE/depth AS normalized_position,
               click::DOUBLE/impression AS observed_ctr,
               user_id=0 AS anonymous_user
        FROM fact_sponsored_search
        """
    )
    connection.execute(
        "CREATE OR REPLACE VIEW mart_model_context AS SELECT * FROM mart_ctr_features"
    )
    connection.execute(
        "CREATE OR REPLACE VIEW mart_experiment_context AS SELECT * FROM mart_model_context "
        "WHERE data_role='experiment_context_pool'"
    )
    connection.execute(
        """
        CREATE OR REPLACE VIEW mart_reporting AS
        SELECT data_role, depth, position, count(*) row_count, sum(impression) impressions,
               sum(click) clicks, sum(click)::DOUBLE/sum(impression) ctr
        FROM fact_sponsored_search GROUP BY data_role, depth, position
        """
    )
    profile_row = connection.execute(
        """
        SELECT count(*) row_count, sum(impression) impressions, sum(click) clicks,
               sum(click)::DOUBLE/sum(impression) global_ctr,
               sum(user_id=0)::DOUBLE/count(*) anonymous_user_rate
        FROM fact_sponsored_search
        """
    ).fetchone()
    role_counts = dict(
        connection.execute(
            "SELECT data_role, count(*) FROM fact_sponsored_search GROUP BY data_role"
        ).fetchall()
    )
    setting_crossings = scalar(
        connection,
        "SELECT count(*) FROM (SELECT setting_hash FROM fact_sponsored_search "
        "GROUP BY setting_hash HAVING count(DISTINCT data_role)>1)",
    )
    distributions = connection.execute(
        "SELECT depth, position, count(*) row_count, sum(impression) impressions, sum(click) clicks "
        "FROM fact_sponsored_search GROUP BY depth, position ORDER BY depth, position"
    ).fetchdf()
    distribution_path = ROOT / "artifacts/qualification/kdd_depth_position_distribution.csv"
    distributions.to_csv(distribution_path, index=False)
    unique_counts = {
        "ad": scalar(connection, "SELECT count(*) FROM dim_ad"),
        "advertiser": scalar(connection, "SELECT count(*) FROM dim_advertiser"),
        "display_url": scalar(connection, "SELECT count(*) FROM dim_display_url"),
        "query": scalar(connection, "SELECT count(*) FROM dim_query"),
        "user": scalar(connection, "SELECT count(*) FROM dim_user"),
    }
    lookup_counts = {
        "query": scalar(connection, "SELECT count(*) FROM lookup_query_tokens"),
        "keyword": scalar(connection, "SELECT count(*) FROM lookup_keyword_tokens"),
        "title": scalar(connection, "SELECT count(*) FROM lookup_title_tokens"),
        "description": scalar(connection, "SELECT count(*) FROM lookup_description_tokens"),
        "user_profile": scalar(connection, "SELECT count(*) FROM lookup_user_profile"),
    }
    connection.execute("CHECKPOINT")
    connection.close()
    profile = {
        "phase": "P04_DATA_QUALITY",
        "generated_at": datetime.now(UTC).isoformat(),
        "official_kdd": True,
        "rows": int(profile_row[0]),
        "impressions": int(profile_row[1]),
        "clicks": int(profile_row[2]),
        "global_ctr": float(profile_row[3]),
        "unique_ad_id": unique_counts["ad"],
        "unique_advertiser_id": unique_counts["advertiser"],
        "unique_display_url": unique_counts["display_url"],
        "unique_query_id": unique_counts["query"],
        "unique_user_id": unique_counts["user"],
        "anonymous_user_rate": float(profile_row[4]),
        "role_counts": role_counts,
        "setting_groups_crossing_roles": setting_crossings,
        "lookup_rows": lookup_counts,
        "database_path": str(DATABASE.relative_to(ROOT)),
        "database_bytes": DATABASE.stat().st_size,
        "database_sha256": sha256_file(DATABASE),
        "disk_free_before_bytes": free_before,
        "disk_free_after_bytes": shutil.disk_usage(ROOT).free,
        "elapsed_seconds": time.perf_counter() - started,
        "pass": setting_crossings == 0 and int(profile_row[0]) == 149_639_105,
    }
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    QUALITY.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    lineage = {
        "phase": "P04_DATA_QUALITY",
        "generated_at": datetime.now(UTC).isoformat(),
        "canonical_engine": f"DuckDB {duckdb.__version__}",
        "sqlite_boundary": "data/processed/aix_page.sqlite is retained as synthetic/test-only historical output",
        "edges": [
            "official track2.zip -> validated partitioned Parquet",
            "validated Parquet -> fact_sponsored_search",
            "fact_sponsored_search -> dim_ad/dim_advertiser/dim_display_url/dim_query/dim_user",
            "fact_sponsored_search -> mart_ctr_features -> mart_model_context",
            "mart_model_context -> mart_experiment_context/mart_reporting",
        ],
        "database_sha256": profile["database_sha256"],
    }
    LINEAGE.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))
    if not profile["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
