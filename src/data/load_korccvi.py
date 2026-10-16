"""data/raw/korccvi/Multimodal 아래 KorCCVi v2 배포본을 읽는 함수들.

모든 함수는 파일 내용을 그대로 dict/list로 돌려준다. 여기서는 정제를 하지
않는다 — 그건 preprocess.py의 일이다.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

# manifest에 적힌 경로("data/transcripts/...")의 기준이 되는 루트가 Multimodal/ 이다.
MULTIMODAL_ROOT = Path(
    os.environ.get("KORCCVI_MULTIMODAL_ROOT", "data/raw/korccvi/Multimodal")
)
DATA_ROOT = MULTIMODAL_ROOT / "data"

CALL_SPLITS = ("train", "val", "test")


def call_manifest_path(split: str) -> Path:
    """{"train", "val", "test", "master"} 중 한 split의 통화 단위 manifest 경로."""
    return DATA_ROOT / f"{split}_manifest.jsonl"


def segment_manifest_path(split: str) -> Path:
    """한 split의 증강되지 않은 원본 구간 단위 manifest 경로."""
    return DATA_ROOT / f"{split}_segment_manifest.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    """.jsonl 파일을 dict 리스트로 읽는다. 빈 줄은 건너뛴다."""
    records = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_call_manifest(split: str) -> list[dict]:
    """통화 레코드: call_id, audio_filepath, transcript_filepath, label."""
    return read_jsonl(call_manifest_path(split))


def load_segment_manifest(split: str) -> list[dict]:
    """구간 레코드: call_id, segment_id, start, end, text, label, 피처 경로들."""
    return read_jsonl(segment_manifest_path(split))


def load_all_call_splits() -> dict[str, list[dict]]:
    """{"train": [...], "val": [...], "test": [...]} 형태의 통화 manifest 묶음."""
    return {split: load_call_manifest(split) for split in CALL_SPLITS}


def load_transcript(transcript_filepath: str) -> dict:
    """manifest에 저장된 경로로 CLOVA Speech 전사본 json 하나를 읽는다."""
    with (MULTIMODAL_ROOT / transcript_filepath).open(encoding="utf-8") as transcript_file:
        return json.load(transcript_file)


def segment_texts(transcript: dict) -> list[str]:
    """파일에 적힌 순서대로의 구간 텍스트. 구간이 없으면 빈 리스트."""
    return [segment.get("text", "") for segment in transcript.get("segments", [])]


def load_fss_source_mapping() -> list[dict]:
    """fss_source_file_mapping.csv의 행들 (FSS 원본 녹음 <-> 피싱 통화 대응).

    컬럼: manifest_file, fss_source_file, match_type, local_duration_s,
    source_duration_s, note. 이 파일은 UTF-8 BOM으로 시작한다.
    """
    mapping_path = DATA_ROOT / "fss_source_file_mapping.csv"
    with mapping_path.open(encoding="utf-8-sig", newline="") as mapping_file:
        return list(csv.DictReader(mapping_file))


def call_id_from_manifest_file(manifest_file: str) -> str:
    """'vishing_12.wav' -> 'vishing_12' (매핑 행은 오디오 파일명을 키로 쓴다)."""
    return Path(manifest_file).stem
