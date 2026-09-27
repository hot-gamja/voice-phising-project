"""Readers for the KorCCVi v2 release under data/raw/korccvi/Multimodal.

All functions return plain dicts/lists straight from the files. No cleaning
happens here; that belongs to preprocess.py.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

# Multimodal/ is the root that manifest paths ("data/transcripts/...") are relative to.
MULTIMODAL_ROOT = Path(
    os.environ.get("KORCCVI_MULTIMODAL_ROOT", "data/raw/korccvi/Multimodal")
)
DATA_ROOT = MULTIMODAL_ROOT / "data"

CALL_SPLITS = ("train", "val", "test")


def call_manifest_path(split: str) -> Path:
    """Path of the call-level manifest for split in {"train", "val", "test", "master"}."""
    return DATA_ROOT / f"{split}_manifest.jsonl"


def segment_manifest_path(split: str) -> Path:
    """Path of the plain (non-augmented) segment-level manifest for a split."""
    return DATA_ROOT / f"{split}_segment_manifest.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    """Read a .jsonl file into a list of dicts. Blank lines are skipped."""
    records = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_call_manifest(split: str) -> list[dict]:
    """Call records: call_id, audio_filepath, transcript_filepath, label."""
    return read_jsonl(call_manifest_path(split))


def load_segment_manifest(split: str) -> list[dict]:
    """Segment records: call_id, segment_id, start, end, text, label, feature paths."""
    return read_jsonl(segment_manifest_path(split))


def load_all_call_splits() -> dict[str, list[dict]]:
    """{"train": [...], "val": [...], "test": [...]} call manifests."""
    return {split: load_call_manifest(split) for split in CALL_SPLITS}


def load_transcript(transcript_filepath: str) -> dict:
    """Load one CLOVA Speech transcript json given the path stored in a manifest."""
    with (MULTIMODAL_ROOT / transcript_filepath).open(encoding="utf-8") as transcript_file:
        return json.load(transcript_file)


def segment_texts(transcript: dict) -> list[str]:
    """Segment texts in file order. Empty list when the transcript has no segments."""
    return [segment.get("text", "") for segment in transcript.get("segments", [])]


def load_fss_source_mapping() -> list[dict]:
    """Rows of fss_source_file_mapping.csv (FSS recording <-> vishing call).

    Columns: manifest_file, fss_source_file, match_type, local_duration_s,
    source_duration_s, note. The file starts with a UTF-8 BOM.
    """
    mapping_path = DATA_ROOT / "fss_source_file_mapping.csv"
    with mapping_path.open(encoding="utf-8-sig", newline="") as mapping_file:
        return list(csv.DictReader(mapping_file))


def call_id_from_manifest_file(manifest_file: str) -> str:
    """'vishing_12.wav' -> 'vishing_12' (mapping rows key on the audio filename)."""
    return Path(manifest_file).stem
