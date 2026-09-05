import json
from pathlib import Path

import pytest

from aix_page.utils.state import receipt


def test_receipt_rejects_missing_pass_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        receipt(tmp_path, "P00_BOOTSTRAP", "PASS", [], [tmp_path / "missing"], [], [], "")


def test_phase_receipt_schema(tmp_path: Path) -> None:
    output = tmp_path / "x"
    output.write_text("x")
    path = receipt(tmp_path, "P00_BOOTSTRAP", "PASS", [], [output], ["cmd"], ["test"], "note")
    body = json.loads(path.read_text())
    assert body["status"] == "PASS" and body["output_hashes"]
