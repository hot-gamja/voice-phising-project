"""P02: 원본 전사본으로부터 data/processed/{train,val,test}.jsonl을 생성한다.

실행:  uv run python -m scripts.build_processed
"""

from __future__ import annotations

from src.data.load_korccvi import CALL_SPLITS
from src.data.preprocess import load_processed_split, write_processed_split


def main() -> None:
    for split in CALL_SPLITS:
        output_path = write_processed_split(split)
        rows = load_processed_split(split)
        empty_count = sum(not row["text"] for row in rows)
        print(f"{split:5s} {len(rows):5d} calls  empty={empty_count}  -> {output_path}")


if __name__ == "__main__":
    main()
