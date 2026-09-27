"""P02: build data/processed/{train,val,test}.jsonl from the raw transcripts.

Run:  uv run python -m scripts.build_processed
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
