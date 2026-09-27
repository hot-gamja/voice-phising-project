"""model_text = Head+Tail-truncated call text, cached per split (plan §5, §9).

Training, TF-IDF, and serving all read the same strings from here.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from src.data.preprocess import PROCESSED_DIR, load_processed_split
from src.data.truncate import truncate_head_tail

TOKENIZER_NAME = "klue/roberta-base"


def model_text_path(split: str) -> Path:
    return PROCESSED_DIR / f"{split}_model_text.jsonl"


def build_model_text_split(split: str, tokenizer=None) -> list[dict]:
    """[{call_id, label, text, model_text}] and write the cache file."""
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    rows = load_processed_split(split)
    for row in rows:
        row["model_text"] = truncate_head_tail(row["text"], tokenizer)
    with model_text_path(split).open("w", encoding="utf-8") as cache_file:
        for row in rows:
            cache_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_model_text_split(split: str) -> list[dict]:
    """Read the cache, building it on first use."""
    path = model_text_path(split)
    if not path.exists():
        return build_model_text_split(split)
    with path.open(encoding="utf-8") as cache_file:
        return [json.loads(line) for line in cache_file if line.strip()]
