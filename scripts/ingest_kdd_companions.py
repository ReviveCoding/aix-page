from __future__ import annotations

import json
import shutil
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from aix_page.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/track2.zip"
OUTPUT = ROOT / "data/staged/kdd/lookups"
MANIFEST = ROOT / "data/manifests/kdd_companion_manifest.json"
MEMBERS: dict[str, tuple[str, ...]] = {
    "query": ("track2/queryid_tokensid.txt", "id", "tokens"),
    "keyword": ("track2/purchasedkeywordid_tokensid.txt", "id", "tokens"),
    "title": ("track2/titleid_tokensid.txt", "id", "tokens"),
    "description": ("track2/descriptionid_tokensid.txt", "id", "tokens"),
    "user_profile": ("track2/userid_profile.txt", "id", "gender", "age"),
}


def invalid_handler(target: list[dict[str, object]]):
    def handle(row: pacsv.InvalidRow) -> str:
        target.append({"number": row.number, "text": row.text[:1000]})
        return "skip"

    return handle


def main() -> None:
    disk_before = shutil.disk_usage(ROOT).free
    if disk_before - 6 * 1024**3 < 50 * 1024**3:
        raise RuntimeError("companion projection violates storage floor")
    results: dict[str, object] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name, definition in MEMBERS.items():
            member, *columns = definition
            destination = OUTPUT / name
            destination.mkdir(parents=True, exist_ok=True)
            bad_rows: list[dict[str, object]] = []

            types = {columns[0]: pa.uint64(), **{column: pa.string() for column in columns[1:]}}
            started = time.perf_counter()
            count = 0
            parts = []
            with archive.open(member) as source:
                reader = pacsv.open_csv(
                    source,
                    read_options=pacsv.ReadOptions(
                        use_threads=True,
                        block_size=64 * 1024 * 1024,
                        column_names=columns,
                    ),
                    parse_options=pacsv.ParseOptions(
                        delimiter="\t", invalid_row_handler=invalid_handler(bad_rows)
                    ),
                    convert_options=pacsv.ConvertOptions(
                        column_types=types, strings_can_be_null=False
                    ),
                )
                for batch_number, batch in enumerate(reader):
                    table = pa.Table.from_batches([batch])
                    part = destination / f"part-{batch_number:05d}.parquet"
                    if part.exists():
                        if pq.read_metadata(part).num_rows != len(table):
                            raise RuntimeError(f"resume mismatch: {part}")
                    else:
                        pq.write_table(table, part, compression="zstd", compression_level=5)
                    count += len(table)
                    parts.append(
                        {
                            "path": str(part.relative_to(ROOT)),
                            "rows": len(table),
                            "bytes": part.stat().st_size,
                            "sha256": sha256_file(part),
                        }
                    )
            results[name] = {
                "member": member,
                "columns": columns,
                "rows": count,
                "invalid_rows": bad_rows,
                "parts": parts,
                "elapsed_seconds": time.perf_counter() - started,
                "pass": count > 0 and not bad_rows,
            }
            print(name, count, "rows", flush=True)
    manifest = {
        "phase": "P03_DATA_INGESTION",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(ARCHIVE.relative_to(ROOT)),
        "retrieval_boundary": "official KDD companion members",
        "lookups": results,
        "disk_free_before_bytes": disk_before,
        "disk_free_after_bytes": shutil.disk_usage(ROOT).free,
        "pass": all(bool(value["pass"]) for value in results.values()),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"pass": manifest["pass"], "disk_free_after_bytes": manifest["disk_free_after_bytes"]},
            indent=2,
        )
    )
    if not manifest["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
