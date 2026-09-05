from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from aix_page.constants import KDD_COLUMNS
from aix_page.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/track2.zip"
TRAINING_MEMBER = "track2/training.txt"
OUTPUT_ROOT = ROOT / "data/staged/kdd/fact_sponsored_search"
QUARANTINE = ROOT / "data/quarantine/kdd_training_rows.tsv"
CHECKPOINT = ROOT / "artifacts/development/kdd_ingestion_checkpoint.json"
MANIFEST = ROOT / "data/manifests/kdd_ingestion_manifest.json"
EXPECTED_ARCHIVE_SHA = "e1fbb172130ba9ae55593f74e5b6bfdd74e1e6046a0b540ffb6b07dd087f5686"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if sha256_file(ARCHIVE) != EXPECTED_ARCHIVE_SHA:
        raise RuntimeError("official KDD archive SHA-256 contract failed")
    disk_before = shutil.disk_usage(ROOT).free
    projected_bytes = 12 * 1024**3
    if disk_before - projected_bytes < 50 * 1024**3:
        raise RuntimeError("storage projection violates 50 GiB safety floor")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
    malformed: list[tuple[int | None, str, str]] = []

    def invalid_row(row: pacsv.InvalidRow) -> str:
        malformed.append((row.number, "FIELD_COUNT_NOT_12", row.text[:1000]))
        return "skip"

    schema = {
        "Click": pa.int64(),
        "Impression": pa.int64(),
        "Depth": pa.int64(),
        "Position": pa.int64(),
        **{
            column: pa.uint64()
            for column in KDD_COLUMNS
            if column not in {"Click", "Impression", "Depth", "Position"}
        },
    }
    read_options = pacsv.ReadOptions(
        use_threads=True,
        block_size=args.batch_bytes,
        column_names=list(KDD_COLUMNS),
    )
    parse_options = pacsv.ParseOptions(delimiter="\t", invalid_row_handler=invalid_row)
    convert_options = pacsv.ConvertOptions(column_types=schema, strings_can_be_null=False)
    started = time.perf_counter()
    total_rows = 0
    total_clicks = 0
    total_impressions = 0
    valid_rows = 0
    domain_invalid = 0
    parts: list[dict[str, object]] = []

    with zipfile.ZipFile(ARCHIVE) as archive, archive.open(TRAINING_MEMBER) as source:
        reader = pacsv.open_csv(
            source,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        for batch_number, batch in enumerate(reader):
            table = pa.Table.from_batches([batch])
            if args.max_rows is not None and total_rows + len(table) > args.max_rows:
                table = table.slice(0, args.max_rows - total_rows)
            arrays = {name: table[name].to_numpy(zero_copy_only=False) for name in KDD_COLUMNS}
            valid = (
                (arrays["Impression"] > 0)
                & (arrays["Click"] >= 0)
                & (arrays["Click"] <= arrays["Impression"])
                & (arrays["Depth"] > 0)
                & (arrays["Position"] > 0)
                & (arrays["Position"] <= arrays["Depth"])
            )
            invalid_indices = np.flatnonzero(~valid)
            domain_invalid += len(invalid_indices)
            for index in invalid_indices[:10_000]:
                malformed.append(
                    (
                        total_rows + int(index) + 1,
                        "DOMAIN_INVARIANT_FAILED",
                        "\t".join(str(int(arrays[column][index])) for column in KDD_COLUMNS),
                    )
                )
            clean = table.filter(pa.array(valid))
            part = OUTPUT_ROOT / f"part-{batch_number:05d}.parquet"
            if part.exists():
                existing = pq.read_metadata(part)
                if existing.num_rows != len(clean):
                    raise RuntimeError(f"resume part row mismatch: {part}")
            else:
                pq.write_table(clean, part, compression="zstd", compression_level=3)
            part_rows = len(clean)
            valid_rows += part_rows
            total_clicks += int(pc.sum(clean["Click"]).as_py() or 0)
            total_impressions += int(pc.sum(clean["Impression"]).as_py() or 0)
            total_rows += len(table)
            parts.append(
                {
                    "path": str(part.relative_to(ROOT)),
                    "rows": part_rows,
                    "bytes": part.stat().st_size,
                    "sha256": sha256_file(part),
                }
            )
            write_json(
                CHECKPOINT,
                {
                    "status": "IN_PROGRESS",
                    "batch_number": batch_number,
                    "source_rows_seen": total_rows,
                    "valid_rows_written": valid_rows,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            if args.max_rows is not None and total_rows >= args.max_rows:
                break

    with QUARANTINE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["row_number", "reason", "raw"])
        writer.writerows(malformed)
    elapsed = time.perf_counter() - started
    manifest = {
        "phase": "P03_DATA_INGESTION",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(ARCHIVE.relative_to(ROOT)),
        "source_sha256": EXPECTED_ARCHIVE_SHA,
        "source_member": TRAINING_MEMBER,
        "official_kdd": True,
        "max_rows": args.max_rows,
        "source_rows_seen": total_rows,
        "valid_rows": valid_rows,
        "quarantined_rows": len(malformed),
        "domain_invalid_rows": domain_invalid,
        "clicks": total_clicks,
        "impressions": total_impressions,
        "global_ctr": total_clicks / total_impressions,
        "parts": parts,
        "elapsed_seconds": elapsed,
        "rows_per_second": total_rows / elapsed,
        "disk_free_before_bytes": disk_before,
        "disk_free_after_bytes": shutil.disk_usage(ROOT).free,
        "safety_floor_bytes": 50 * 1024**3,
        "complete": args.max_rows is None,
    }
    manifest["pass"] = (
        valid_rows > 0
        and len(malformed) == 0
        and manifest["disk_free_after_bytes"] >= manifest["safety_floor_bytes"]
    )
    write_json(MANIFEST, manifest)
    write_json(
        CHECKPOINT,
        {
            "status": "COMPLETE",
            "source_rows_seen": total_rows,
            "valid_rows_written": valid_rows,
            "manifest_sha256": sha256_file(MANIFEST),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "parts"}, indent=2))
    if not manifest["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
