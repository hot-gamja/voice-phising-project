"""P01 데이터셋 감사: 공식 split에 대한 정합성 검사 + 집계 통계.

실행:  uv run python -m scripts.audit_korccvi
experiments/p01_audit_report.md를 쓴다. call_id만 출력하며, Test 텍스트는 절대 출력하지 않는다.
"""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

from src.data.load_korccvi import (
    CALL_SPLITS,
    load_all_call_splits,
    load_fss_source_mapping,
    load_segment_manifest,
    load_transcript,
    segment_texts,
)
from src.data.split_check import (
    EXPECTED_TOTAL_CALLS,
    call_id_overlaps,
    duplicate_call_ids_within_split,
    duplicate_transcripts,
    invalid_labels,
    segment_call_id_mismatch,
    source_overlap_report,
    total_calls,
)

REPORT_PATH = Path("experiments/p01_audit_report.md")
LABEL_NAME = {0: "non_vishing", 1: "vishing"}


def describe(values: list[int]) -> str:
    """한 줄로 min/median/mean/max 표시."""
    if not values:
        return "n=0"
    return (
        f"n={len(values)} min={min(values)} median={statistics.median(values):.0f} "
        f"mean={statistics.mean(values):.0f} max={max(values)}"
    )


def main() -> None:
    lines: list[str] = ["# P01 KorCCVi v2 Audit Report", ""]
    call_splits = load_all_call_splits()

    # --- 1-4: call_id 중복 없음 및 총계 ---
    lines += ["## Split integrity (plan §8-A checks 1-5)", ""]
    for pair, shared in call_id_overlaps(call_splits).items():
        lines.append(f"- call_id overlap {pair}: {len(shared)}")
    for split, records in call_splits.items():
        lines.append(f"- duplicate call_id within {split}: {duplicate_call_ids_within_split(records)}")
    lines.append(f"- total calls: {total_calls(call_splits)} (expected {EXPECTED_TOTAL_CALLS})")

    # --- 5: segment manifest과 call manifest 일치 확인 ---
    segment_splits = {split: load_segment_manifest(split) for split in CALL_SPLITS}
    for split in CALL_SPLITS:
        mismatch = segment_call_id_mismatch(call_splits[split], segment_splits[split])
        lines.append(
            f"- {split}: segment-only call_ids={len(mismatch['segment_only'])}, "
            f"manifest-only call_ids={len(mismatch['manifest_only'])}, "
            f"segments={len(segment_splits[split])}"
        )
    for split in CALL_SPLITS:
        lines.append(f"- invalid labels in {split}: {len(invalid_labels(call_splits[split]))}")
    lines.append("")

    # --- split별, 클래스별 개수 ---
    lines += ["## Counts", "", "| split | non_vishing | vishing | total |", "|---|---:|---:|---:|"]
    for split, records in call_splits.items():
        label_counts = Counter(record["label"] for record in records)
        lines.append(f"| {split} | {label_counts[0]} | {label_counts[1]} | {len(records)} |")
    lines.append("")

    # --- 전사본 단위 품질 및 길이 통계 (집계만) ---
    lines += ["## Transcript quality and length (aggregate)", ""]
    call_text_by_id: dict[str, str] = {}
    empty_calls: list[tuple[str, str]] = []
    asr_versions: Counter = Counter()
    speaker_field_present = Counter()
    joined_equals_text_field = Counter()
    char_lengths = {split: {0: [], 1: []} for split in CALL_SPLITS}
    segment_counts = {split: {0: [], 1: []} for split in CALL_SPLITS}
    for split, records in call_splits.items():
        for record in records:
            transcript = load_transcript(record["transcript_filepath"])
            texts = segment_texts(transcript)
            joined = " ".join(text.strip() for text in texts if text.strip())
            call_text_by_id[record["call_id"]] = joined
            if not joined:
                empty_calls.append((split, record["call_id"]))
            asr_versions[transcript.get("version", "?")] += 1
            speaker_field_present[all("speaker" in segment for segment in transcript.get("segments", []))] += 1
            joined_equals_text_field["".join(joined.split()) == "".join(transcript.get("text", "").split())] += 1
            char_lengths[split][record["label"]].append(len(joined))
            segment_counts[split][record["label"]].append(len(texts))

    lines.append(f"- empty transcripts: {len(empty_calls)} {empty_calls[:10]}")
    lines.append(f"- ASR version strings: {len(asr_versions)} distinct")
    for version, count in asr_versions.most_common(3):
        lines.append(f"  - {count}x `{version}`")
    lines.append(f"- every segment has `speaker` field: {dict(speaker_field_present)}")
    lines.append(f"- joined segments == top-level `text` (whitespace-insensitive): {dict(joined_equals_text_field)}")
    split_of_call = {record["call_id"]: split for split, records in call_splits.items() for record in records}
    duplicates = duplicate_transcripts(call_text_by_id)
    lines.append(f"- duplicate transcript groups (whitespace-insensitive): {len(duplicates)}")
    for group in list(duplicates.values())[:20]:
        lines.append(f"  - {[(call_id, split_of_call[call_id]) for call_id in group]}")
    lines.append("")
    lines += ["### Character length of joined transcript", ""]
    for split in CALL_SPLITS:
        for label in (0, 1):
            lines.append(f"- {split} {LABEL_NAME[label]}: {describe(char_lengths[split][label])}")
    lines += ["", "### Segments per call", ""]
    for split in CALL_SPLITS:
        for label in (0, 1):
            lines.append(f"- {split} {LABEL_NAME[label]}: {describe(segment_counts[split][label])}")
    lines.append("")

    # --- 6: FSS source overlap (보고만) ---
    lines += ["## FSS source overlap (plan §8-A report 6)", ""]
    source_report = source_overlap_report(call_splits, load_fss_source_mapping())
    lines.append(f"- mapping rows: {source_report['mapping_rows']}")
    lines.append(f"- distinct source recordings: {source_report['distinct_sources']}")
    lines.append(f"- match_type: {dict(source_report['match_type_counts'])}")
    lines.append(f"- vishing calls without mapping row: {source_report['vishing_calls_without_mapping']}")
    lines.append(f"- mapping rows without a call: {[row['manifest_file'] for row in source_report['mapping_rows_without_call']]}")
    crossing = source_report["sources_crossing_splits"]
    lines.append(f"- **sources whose calls cross splits: {len(crossing)}**")
    for source, calls in crossing.items():
        lines.append(f"  - `{source[:60]}` → {calls}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
