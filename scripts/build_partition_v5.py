from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data/processed/aix_page.duckdb"
REPORT = ROOT / "artifacts/qualification/partition_v7_manifest.json"


def main() -> None:
    connection = duckdb.connect(str(DATABASE))
    connection.execute("SET threads=8")
    connection.execute(
        """
        CREATE OR REPLACE VIEW fact_sponsored_search_v7 AS
        WITH keyed AS (
          SELECT * EXCLUDE(data_role),
                 hash(user_id, ad_id, advertiser_id, display_url_id, query_id,
                      keyword_id, title_id, description_id, position, depth,
                      'partition_v7') AS setting_hash_v7
          FROM fact_sponsored_search
        )
        SELECT *,
          CASE
            WHEN impression > 1000 OR hash(setting_hash_v7, 'role_v7') % 1000 < 50
              THEN 'challenge_pool'
            WHEN hash(setting_hash_v7, 'role_v7') % 1000 < 550 THEN 'model_train'
            WHEN hash(setting_hash_v7, 'role_v7') % 1000 < 650 THEN 'model_validation'
            WHEN hash(setting_hash_v7, 'role_v7') % 1000 < 750 THEN 'model_calibration'
            WHEN hash(setting_hash_v7, 'role_v7') % 1000 < 850 THEN 'model_locked_test'
            ELSE 'experiment_context_pool'
          END AS data_role
        FROM keyed
        """
    )
    crossings = int(
        connection.execute(
            """
            SELECT count(*) FROM (
              SELECT setting_hash_v7 FROM fact_sponsored_search_v7
              GROUP BY setting_hash_v7 HAVING count(DISTINCT data_role)>1
            )
            """
        ).fetchone()[0]
    )
    if crossings:
        raise RuntimeError(
            f"extreme-support routing split {crossings} setting groups; "
            "a group-level routing table is required"
        )
    rows = connection.execute(
        """
        SELECT data_role, count(*) row_count, sum(impression) impressions,
               sum(click) clicks, sum(click)::DOUBLE/sum(impression) ctr,
               max(impression) max_impression
        FROM fact_sponsored_search_v7 GROUP BY data_role ORDER BY data_role
        """
    ).fetchdf()
    connection.execute("CHECKPOINT")
    connection.close()
    report = {
        "phase": "P05_FEATURE_PIPELINE",
        "generated_at": datetime.now(UTC).isoformat(),
        "partition_version": "v7",
        "salt": "partition_v7/role_v7",
        "setting_key": [
            "user_id",
            "ad_id",
            "advertiser_id",
            "display_url_id",
            "query_id",
            "keyword_id",
            "title_id",
            "description_id",
            "position",
            "depth",
        ],
        "extreme_support_rule": "impression > 1000 routes to challenge_pool in addition to the 5% hash challenge allocation; clicks unused",
        "setting_groups_crossing_roles": crossings,
        "roles": rows.to_dict(orient="records"),
        "pass": crossings == 0,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
