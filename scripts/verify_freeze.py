"""Verify the V1 freeze: recompute hashes and compare them to experiments/freeze.json.

Run:  uv run python -m scripts.verify_freeze
Read-only — never writes freeze.json. Exits non-zero if any frozen model artifact
or frozen code file no longer matches what was recorded at freeze time.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze import FREEZE_PATH, sha256_of

MATCH = "MATCH"
MISMATCH = "MISMATCH"
MISSING = "MISSING"


def verdict(path: Path, expected_hash: str) -> tuple[str, str]:
    """Compare one frozen path against its recorded hash.

    Returns (status, detail) where status is MATCH / MISMATCH / MISSING.
    """
    if not path.exists():
        return MISSING, f"no such path: {path}"
    actual_hash = sha256_of(path)
    if actual_hash == expected_hash:
        return MATCH, actual_hash
    return MISMATCH, f"recorded {expected_hash} -> actual {actual_hash}"


def frozen_entries(record: dict) -> list[tuple[str, Path, str]]:
    """Flatten freeze.json into (label, path, expected_hash) rows."""
    entries = [(name, Path(info["path"]), info["sha256_16"])
               for name, info in record["models"].items()]
    entries += [(path, Path(path), expected_hash)
                for path, expected_hash in record["code"].items()]
    return entries


def main() -> None:
    if not FREEZE_PATH.exists():
        raise SystemExit(f"{FREEZE_PATH} not found. Nothing to verify.")
    record = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    entries = frozen_entries(record)

    failures = []
    for label, path, expected_hash in entries:
        status, detail = verdict(path, expected_hash)
        print(f"{status:9} {label:34} {detail}")
        if status != MATCH:
            failures.append(label)

    print(f"\n{len(entries) - len(failures)}/{len(entries)} match "
          f"(frozen at {record['frozen_at']})")
    if failures:
        raise SystemExit(f"freeze verification FAILED: {', '.join(failures)}")


if __name__ == "__main__":
    main()
