"""V1 freeze를 검증한다: 해시를 다시 계산해 experiments/freeze.json과 비교한다.

실행:  uv run python -m scripts.verify_freeze
읽기 전용 — freeze.json을 절대 쓰지 않는다. freeze 시점에 기록된 것과 모델 artifact나
코드 파일이 하나라도 일치하지 않으면 0이 아닌 코드로 종료한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze import FREEZE_PATH, sha256_of

MATCH = "MATCH"
MISMATCH = "MISMATCH"
MISSING = "MISSING"


def verdict(path: Path, expected_hash: str) -> tuple[str, str]:
    """하나의 freeze 대상 경로를 기록된 해시와 비교한다.

    status가 MATCH / MISMATCH / MISSING 중 하나인 (status, detail)을 반환한다.
    """
    if not path.exists():
        return MISSING, f"no such path: {path}"
    actual_hash = sha256_of(path)
    if actual_hash == expected_hash:
        return MATCH, actual_hash
    return MISMATCH, f"recorded {expected_hash} -> actual {actual_hash}"


def frozen_entries(record: dict) -> list[tuple[str, Path, str]]:
    """freeze.json을 (label, path, expected_hash) 행들로 펼친다."""
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
