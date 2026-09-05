"""Phase state and immutable evidence receipts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aix_page.constants import PHASES
from aix_page.utils.hashing import sha256_file


def receipt(
    root: Path,
    phase: str,
    status: str,
    inputs: list[Path],
    outputs: list[Path],
    commands: list[str],
    tests: list[str],
    notes: str,
    config: Path | None = None,
) -> Path:
    if phase not in PHASES or status not in {"PASS", "FAIL", "BLOCKED"}:
        raise ValueError("invalid receipt phase/status")
    missing = [str(p) for p in outputs if not p.exists()]
    if status == "PASS" and missing:
        raise FileNotFoundError(f"PASS outputs missing: {missing}")
    body: dict[str, Any] = {
        "phase": phase,
        "timestamp": datetime.now(UTC).isoformat(),
        "config_hash": sha256_file(config) if config and config.exists() else None,
        "input_hashes": {str(p): sha256_file(p) for p in inputs if p.is_file()},
        "output_hashes": {str(p): sha256_file(p) for p in outputs if p.is_file()},
        "commands": commands,
        "tests_executed": tests,
        "status": status,
        "notes": notes,
    }
    target = root / "artifacts" / "stage_receipts" / f"{phase}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return target


def advance(root: Path, completed_phase: str, next_phase: str) -> None:
    state_path = root / "artifacts" / "EXECUTION_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if completed_phase not in state["completed_phases"]:
        state["completed_phases"].append(completed_phase)
    state["current_phase"] = next_phase
    state["last_receipt"] = f"artifacts/stage_receipts/{completed_phase}.json"
    state["updated_at"] = datetime.now(UTC).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
