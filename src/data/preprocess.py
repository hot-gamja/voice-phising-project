"""Minimal preprocessing (plan §7): segments -> one call text, whitespace only.

No number/keyword removal, no morphological analysis. The joined text is the
full call; truncation for the model happens later in truncate.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.load_korccvi import load_call_manifest, load_transcript, segment_texts

PROCESSED_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    """Trim and collapse runs of whitespace to one space. Nothing else."""
    return " ".join(text.split())


def call_text_from_transcript(transcript: dict) -> str:
    """Join segment texts in order with a single space, no speaker tags (plan §5)."""
    cleaned_segments = [clean_text(text) for text in segment_texts(transcript)]
    return " ".join(segment for segment in cleaned_segments if segment)


def build_processed_split(split: str) -> list[dict]:
    """[{call_id, label, text}] for one official split, in manifest order."""
    rows = []
    for record in load_call_manifest(split):
        transcript = load_transcript(record["transcript_filepath"])
        rows.append({
            "call_id": record["call_id"],
            "label": record["label"],
            "text": call_text_from_transcript(transcript),
        })
    return rows


def write_processed_split(split: str) -> Path:
    """Write data/processed/{split}.jsonl and return its path."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in build_processed_split(split):
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def load_processed_split(split: str) -> list[dict]:
    """Read back data/processed/{split}.jsonl."""
    with (PROCESSED_DIR / f"{split}.jsonl").open(encoding="utf-8") as processed_file:
        return [json.loads(line) for line in processed_file if line.strip()]
