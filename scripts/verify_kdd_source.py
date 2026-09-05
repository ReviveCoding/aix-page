from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/track2.zip"
EXPECTED_BYTES = 2_882_914_995
EXPECTED_SHA256 = "E1FBB172130BA9AE55593F74E5B6BFDD74E1E6046A0B540FFB6B07DD087F5686"
EXPECTED_MEMBERS = {
    "track2/training.txt",
    "track2/queryid_tokensid.txt",
    "track2/purchasedkeywordid_tokensid.txt",
    "track2/titleid_tokensid.txt",
    "track2/descriptionid_tokensid.txt",
    "track2/userid_profile.txt",
}


def main() -> None:
    disk_before = shutil.disk_usage(ROOT).free
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    sha256 = digest.hexdigest().upper()
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = {
            item.filename: {
                "uncompressed_bytes": item.file_size,
                "compressed_bytes": item.compress_size,
                "crc32": f"{item.CRC:08X}",
            }
            for item in archive.infolist()
        }
        bad_member = archive.testzip()
    size = ARCHIVE.stat().st_size
    missing = sorted(EXPECTED_MEMBERS - members.keys())
    report = {
        "phase": "P02_DATA_ACCESS",
        "generated_at": datetime.now(UTC).isoformat(),
        "competition": "kddcup2012-track2",
        "source": "official Kaggle competition",
        "retrieval_method": "official Kaggle web UI",
        "retrieval_timestamp_from_file_mtime_utc": datetime.fromtimestamp(
            ARCHIVE.stat().st_mtime, UTC
        ).isoformat(),
        "archive_path": str(ARCHIVE.relative_to(ROOT)),
        "archive_bytes": size,
        "archive_sha256": sha256,
        "expected_bytes": EXPECTED_BYTES,
        "expected_sha256": EXPECTED_SHA256,
        "byte_size_pass": size == EXPECTED_BYTES,
        "sha256_pass": sha256 == EXPECTED_SHA256,
        "zip_valid": zipfile.is_zipfile(ARCHIVE),
        "full_crc_pass": bad_member is None,
        "first_bad_crc_member": bad_member,
        "expected_members_pass": not missing,
        "missing_expected_members": missing,
        "members": members,
        "disk_free_before_bytes": disk_before,
        "disk_free_after_bytes": shutil.disk_usage(ROOT).free,
    }
    report["pass"] = all(
        report[key]
        for key in [
            "byte_size_pass",
            "sha256_pass",
            "zip_valid",
            "full_crc_pass",
            "expected_members_pass",
        ]
    )
    output = ROOT / "data/manifests/kdd_source_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "members"}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
