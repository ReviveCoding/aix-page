from pathlib import Path

from aix_page.ingestion.parser import parse_line, stream_tsv


def test_exact_12_field_parser() -> None:
    row = "1\t2\t3\t4\t5\t2\t1\t8\t9\t10\t11\t0\n"
    result = parse_line(row)
    assert result.reason is None and result.values is not None and len(result.values) == 12


def test_malformed_and_domain_quarantine(tmp_path: Path) -> None:
    source = tmp_path / "rows.tsv"
    source.write_text("1\t2\n3\t2\t3\t4\t5\t2\t1\t8\t9\t10\t11\t12\n", encoding="utf-8")
    rows = list(stream_tsv(source, tmp_path / "bad.tsv"))
    assert rows == []
    assert "FIELD_COUNT_NOT_12" in (tmp_path / "bad.tsv").read_text()


def test_click_must_not_exceed_impression() -> None:
    assert parse_line("3\t2\t3\t4\t5\t2\t1\t8\t9\t10\t11\t12").reason == "CLICK_OUT_OF_RANGE"
