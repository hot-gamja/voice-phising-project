"""P01 직접 읽기: 사람이 읽을 Train 전용 통화 전사본을 추출한다.

실행:  uv run python -m scripts.dump_train_reading_samples
data/reading/train_samples.md를 쓴다 (gitignore됨: KorCCVi 콘텐츠).
클래스별로 가장 짧은 1/3에서 5개, 가장 긴 1/3에서 5개를 골라
양쪽 끝을 다 보게 한다. Validation과 Test는 절대 건드리지 않는다.
"""

from __future__ import annotations

import random
from pathlib import Path

from src.data.load_korccvi import load_call_manifest, load_transcript

OUTPUT_PATH = Path("data/reading/train_samples.md")
PER_END = 5
SEED = 42
LABEL_NAME = {0: "정상 (non_vishing)", 1: "피싱 (vishing)"}


def call_as_dialogue(transcript: dict) -> str:
    """구간마다 한 줄, CLOVA 화자(speaker) 라벨을 앞에 붙인다."""
    lines = []
    for segment in transcript.get("segments", []):
        speaker = segment.get("speaker", {}).get("label", "?")
        lines.append(f"[{speaker}] {segment.get('text', '').strip()}")
    return "\n".join(lines)


def main() -> None:
    random_generator = random.Random(SEED)
    records = load_call_manifest("train")
    transcripts = {record["call_id"]: load_transcript(record["transcript_filepath"]) for record in records}

    chosen = []
    for label in (0, 1):
        same_label = [record for record in records if record["label"] == label]
        same_label.sort(key=lambda record: len(transcripts[record["call_id"]].get("text", "")))
        third = len(same_label) // 3
        chosen += random_generator.sample(same_label[:third], PER_END)
        chosen += random_generator.sample(same_label[-third:], PER_END)

    parts = ["# P01 Train 직접 읽기 샘플 (seed 42)", "",
             "메모할 것: 왜 피싱/정상인지, ASR 오류, 금융 표현이 많은 정상 통화, 애매한 피싱, 내용과 무관한 형식 차이.", ""]
    for index, record in enumerate(chosen, 1):
        transcript = transcripts[record["call_id"]]
        parts.append(f"## {index}. {record['call_id']} — {LABEL_NAME[record['label']]} — {len(transcript.get('text', ''))}자")
        parts.append("")
        parts.append("```text")
        parts.append(call_as_dialogue(transcript))
        parts.append("```")
        parts.append("")
        parts.append("메모:")
        parts.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {len(chosen)} calls to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
