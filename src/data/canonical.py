"""Canonical V2 rows: V1's {call_id, label, text} plus the dataset a call came from.

V1's data/processed/*.jsonl has no source field, so per-dataset metrics cannot be
computed at all — fatal for a generalization study. These rows add one.

Written as a separate module on a separate output path because
src/data/preprocess.py is sealed by experiments/freeze.json and must stay
byte-identical; its functions are imported, never edited.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.load_korccvi import load_call_manifest, load_transcript
from src.data.preprocess import call_text_from_transcript

CANONICAL_DIR = Path("data/processed/v2")

# Dataset this module reads. External datasets get their own builder later.
DATASET = "korccvi_v2"

# The two upstream recording sources KorCCVi was compiled from, one per class
# (upstream DATA.md §2.1, §2.2). No AI Hub dataSetSn is encoded: upstream names
# the dataset「저품질 전화망 음성인식 데이터」but never states its number, and our
# own plan marks the "#571" guess as inference from domain descriptions only.
SOURCE_FSS = "fss_voice_phishing_guard"
SOURCE_AIHUB = "aihub_lowquality_telephony"


def source_of(record: dict) -> str:
    """Upstream recording source of one call, from a KorCCVi call manifest row.

    Derived from the label on purpose. Within KorCCVi source and label are
    perfectly correlated by construction — every vishing call is FSS and every
    non-vishing call is AI Hub (DATA.md §4) — so this column carries no extra
    information here. That correlation IS the confound V2 measures, and
    recording it explicitly is what makes per-source metrics expressible once
    a third source is added. Other datasets need their own mapping.
    """
    return SOURCE_FSS if record["label"] == 1 else SOURCE_AIHUB


def build_canonical_split(split: str) -> list[dict]:
    """[{call_id, label, text, dataset, source}] for one KorCCVi split, manifest order."""
    rows = []
    for record in load_call_manifest(split):
        transcript = load_transcript(record["transcript_filepath"])
        rows.append({
            "call_id": record["call_id"],
            "label": record["label"],
            "text": call_text_from_transcript(transcript),
            "dataset": DATASET,
            "source": source_of(record),
        })
    return rows


def write_canonical_split(split: str) -> Path:
    """Write data/processed/v2/{split}.jsonl and return its path."""
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CANONICAL_DIR / f"{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in build_canonical_split(split):
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def load_canonical_split(split: str) -> list[dict]:
    """Read back data/processed/v2/{split}.jsonl."""
    with (CANONICAL_DIR / f"{split}.jsonl").open(encoding="utf-8") as canonical_file:
        return [json.loads(line) for line in canonical_file if line.strip()]
