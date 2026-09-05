"""Strict streaming parser for the official KDD Track 2 training schema."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from aix_page.constants import KDD_COLUMNS


@dataclass(frozen=True)
class ParseResult:
    row_number: int
    values: tuple[int, ...] | None
    reason: str | None


def parse_line(line: str, row_number: int = 1) -> ParseResult:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 12:
        return ParseResult(row_number, None, "FIELD_COUNT_NOT_12")
    try:
        values = tuple(int(value) for value in fields)
    except ValueError:
        return ParseResult(row_number, None, "NON_INTEGER_FIELD")
    click, impression = values[0], values[1]
    depth, position = values[5], values[6]
    if impression <= 0:
        return ParseResult(row_number, None, "IMPRESSION_NOT_POSITIVE")
    if click < 0 or click > impression:
        return ParseResult(row_number, None, "CLICK_OUT_OF_RANGE")
    if depth <= 0 or position <= 0 or position > depth:
        return ParseResult(row_number, None, "POSITION_DEPTH_INVALID")
    if any(value < 0 for value in values[2:]):
        return ParseResult(row_number, None, "NEGATIVE_ID")
    return ParseResult(row_number, values, None)


def stream_tsv(path: Path, quarantine_path: Path) -> Iterator[dict[str, int]]:
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("r", encoding="utf-8", newline="") as source,
        quarantine_path.open("w", encoding="utf-8", newline="") as bad,
    ):
        writer = csv.writer(bad, delimiter="\t")
        writer.writerow(["row_number", "reason", "raw"])
        for number, line in enumerate(source, 1):
            result = parse_line(line, number)
            if result.values is None:
                writer.writerow([number, result.reason, line.rstrip("\r\n")])
            else:
                yield dict(zip(KDD_COLUMNS, result.values, strict=True))
