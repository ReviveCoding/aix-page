"""Stable hashing and artifact identities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_mod(modulus: int, *parts: object) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    payload = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
