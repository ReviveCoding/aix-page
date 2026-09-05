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
OUTPUT = ROOT / "data/processed/kdd_features_v7"
MANIFEST = ROOT / "artifacts/qualification/kdd_feature_manifest_v7.json"
KEYS = {
    "ad": ["ad_id"],
    "advertiser": ["advertiser_id"],
    "display_url": ["display_url_id"],
    "query": ["query_id"],
    "user": ["user_id"],
    "position_depth": ["position", "depth"],
    "ad_query": ["ad_id", "query_id"],
}


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.rglob("*.parquet")):
        raise FileExistsError(f"feature output already exists: {OUTPUT}")
    free_before = shutil.disk_usage(ROOT).free
    if free_before - 18 * 1024**3 < 50 * 1024**3:
        raise RuntimeError("feature projection violates 50 GiB floor")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection = duckdb.connect(str(DATABASE))
    connection.execute("SET threads=8")
    connection.execute(f"SET temp_directory='{(ROOT / 'data/processed/duckdb_tmp').as_posix()}'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(
        """
        CREATE OR REPLACE TABLE qualification_base AS
        SELECT *, (hash(setting_hash_v7, 'oof_v7') % 5)::UTINYINT AS oof_fold
        FROM fact_sponsored_search_v7
        WHERE hash(setting_hash_v7, 'qualification_sample_v7') % 1000 <
          CASE data_role
            WHEN 'model_train' THEN 80
            WHEN 'experiment_context_pool' THEN 80
            ELSE 100
          END
        """
    )
    train = "qualification_base WHERE data_role='model_train'"
    for name, columns in KEYS.items():
        group = ", ".join(columns)
        connection.execute(
            f"CREATE OR REPLACE TABLE hist_{name}_total AS "
            f"SELECT {group}, sum(click)::DOUBLE clicks, sum(impression)::DOUBLE impressions, "
            f"count(*) support FROM {train} GROUP BY {group}"
        )
        connection.execute(
            f"CREATE OR REPLACE TABLE hist_{name}_fold AS "
            f"SELECT {group}, oof_fold, sum(click)::DOUBLE clicks, "
            f"sum(impression)::DOUBLE impressions FROM {train} GROUP BY {group}, oof_fold"
        )
    global_clicks, global_impressions = connection.execute(
        f"SELECT sum(click)::DOUBLE, sum(impression)::DOUBLE FROM {train}"
    ).fetchone()
    global_ctr = float(global_clicks / global_impressions)
    joins: list[str] = []
    features: list[str] = []
    for name, columns in KEYS.items():
        condition_total = " AND ".join(f"b.{column}=t_{name}.{column}" for column in columns)
        condition_fold = " AND ".join(f"b.{column}=f_{name}.{column}" for column in columns)
        joins.append(f"LEFT JOIN hist_{name}_total t_{name} ON {condition_total}")
        joins.append(
            f"LEFT JOIN hist_{name}_fold f_{name} ON {condition_fold} "
            f"AND b.oof_fold=f_{name}.oof_fold"
        )
        clicks = (
            f"CASE WHEN b.data_role='model_train' THEN "
            f"coalesce(t_{name}.clicks,0)-coalesce(f_{name}.clicks,0) "
            f"ELSE coalesce(t_{name}.clicks,0) END"
        )
        impressions = (
            f"CASE WHEN b.data_role='model_train' THEN "
            f"coalesce(t_{name}.impressions,0)-coalesce(f_{name}.impressions,0) "
            f"ELSE coalesce(t_{name}.impressions,0) END"
        )
        features.append(
            f"(({clicks}) + 20.0*{global_ctr}) / (({impressions}) + 20.0) AS hist_{name}_ctr"
        )
        features.append(f"coalesce(t_{name}.support,0) AS {name}_support")
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW qualification_features AS
        WITH tokenized AS (
          SELECT b.*,
                 string_split(q.tokens, '|') query_tokens,
                 string_split(k.tokens, '|') keyword_tokens,
                 string_split(ti.tokens, '|') title_tokens,
                 string_split(d.tokens, '|') description_tokens
          FROM qualification_base b
          LEFT JOIN lookup_query_tokens q ON b.query_id=q.query_id
          LEFT JOIN lookup_keyword_tokens k ON b.keyword_id=k.keyword_id
          LEFT JOIN lookup_title_tokens ti ON b.title_id=ti.title_id
          LEFT JOIN lookup_description_tokens d ON b.description_id=d.description_id
        )
        SELECT b.*,
               b.position::DOUBLE/b.depth AS normalized_position,
               b.user_id=0 AS anonymous_user,
               t_query.query_id IS NULL AS cold_query,
               t_ad.ad_id IS NULL AS cold_ad,
               t_advertiser.advertiser_id IS NULL AS cold_advertiser,
               b.user_id=0 OR t_user.user_id IS NULL AS cold_user,
               coalesce(t_query.support,0)<=2 AS tail_query,
               coalesce(t_query.support,0)<=2 OR coalesce(t_ad.support,0)<=2 AS low_support,
               b.depth>=3 AS high_depth,
               b.position=1 OR b.position=b.depth AS position_extreme,
               coalesce(list_count(b.query_tokens),0) query_token_count,
               coalesce(list_count(b.keyword_tokens),0) keyword_token_count,
               coalesce(list_count(b.title_tokens),0) title_token_count,
               coalesce(list_count(b.description_tokens),0) description_token_count,
               coalesce(list_count(list_intersect(b.query_tokens,b.keyword_tokens)),0) query_keyword_overlap,
               coalesce(list_count(list_intersect(b.query_tokens,b.title_tokens)),0) query_title_overlap,
               coalesce(list_count(list_intersect(b.query_tokens,b.description_tokens)),0) query_description_overlap,
               coalesce(list_count(list_intersect(b.query_tokens,b.keyword_tokens)),0)::DOUBLE /
                 greatest(1,coalesce(list_count(list_distinct(list_concat(b.query_tokens,b.keyword_tokens))),0))
                 AS query_keyword_jaccard,
               {", ".join(features)}
        FROM tokenized b
        {" ".join(joins)}
        """
    )
    connection.execute(
        f"COPY (SELECT * EXCLUDE(query_tokens, keyword_tokens, title_tokens, description_tokens) "
        f"FROM qualification_features) TO '{OUTPUT.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY(data_role), OVERWRITE_OR_IGNORE FALSE)"
    )
    counts = dict(
        connection.execute(
            "SELECT data_role, count(*) FROM qualification_base GROUP BY data_role"
        ).fetchall()
    )
    crossing = connection.execute(
        "SELECT count(*) FROM (SELECT setting_hash_v7 FROM qualification_base GROUP BY setting_hash_v7 "
        "HAVING count(DISTINCT data_role)>1)"
    ).fetchone()[0]
    nulls = connection.execute(
        "SELECT sum(hist_ad_ctr IS NULL)+sum(hist_query_ctr IS NULL) FROM qualification_features"
    ).fetchone()[0]
    connection.execute("CHECKPOINT")
    connection.close()
    files = sorted(OUTPUT.rglob("*.parquet"))
    manifest = {
        "phase": "P05_FEATURE_PIPELINE",
        "generated_at": datetime.now(UTC).isoformat(),
        "official_kdd": True,
        "source_rows": 149_639_105,
        "qualification_rows": sum(counts.values()),
        "role_counts": counts,
        "setting_groups_crossing_roles": int(crossing),
        "partition_version": "v7",
        "oof_folds": 5,
        "smoothing_impressions": 20.0,
        "global_training_ctr": global_ctr,
        "historical_features": list(KEYS),
        "token_features": [
            "query_keyword_overlap",
            "query_title_overlap",
            "query_description_overlap",
            "query_keyword_jaccard",
            "token_counts",
        ],
        "challenge_flags": [
            "cold_query",
            "cold_ad",
            "cold_advertiser",
            "cold_user",
            "tail_query",
            "low_support",
            "high_depth",
            "position_extreme",
        ],
        "null_historical_feature_cells": int(nulls),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "disk_free_before_bytes": free_before,
        "disk_free_after_bytes": shutil.disk_usage(ROOT).free,
        "elapsed_seconds": time.perf_counter() - started,
        "pass": int(crossing) == 0 and int(nulls) == 0 and sum(counts.values()) >= 10_000_000,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))
    if not manifest["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
