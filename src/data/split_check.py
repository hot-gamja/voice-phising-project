"""공식 KorCCVi split 무결성 검사 (계획서 §8-A, §6 품질).

검사 1~5는 통과/실패를 가린다. 출처 중복(6)과 중복 대본은 보고만 한다 —
V1에서는 공식 split을 그대로 쓴다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from src.data.load_korccvi import call_id_from_manifest_file

EXPECTED_TOTAL_CALLS = 1417
VALID_LABELS = {0, 1}


def call_ids(call_manifest: list[dict]) -> list[str]:
    return [record["call_id"] for record in call_manifest]


def call_id_overlaps(call_splits: dict[str, list[dict]]) -> dict[str, set[str]]:
    """검사 1~3: 두 split이 공유하는 call_id. 모든 값이 비어 있어야 한다."""
    id_sets = {split: set(call_ids(records)) for split, records in call_splits.items()}
    overlaps = {}
    for split_a, split_b in combinations(id_sets, 2):
        overlaps[f"{split_a}∩{split_b}"] = id_sets[split_a] & id_sets[split_b]
    return overlaps


def duplicate_call_ids_within_split(call_manifest: list[dict]) -> list[str]:
    """한 manifest 안에서 두 번 이상 나타나는 call_id."""
    counts = Counter(call_ids(call_manifest))
    return sorted(call_id for call_id, count in counts.items() if count > 1)


def total_calls(call_splits: dict[str, list[dict]]) -> int:
    """검사 4: 세 split의 합계. EXPECTED_TOTAL_CALLS와 같아야 한다."""
    return sum(len(records) for records in call_splits.values())


def segment_call_id_mismatch(call_manifest: list[dict], segment_manifest: list[dict]) -> dict[str, set[str]]:
    """검사 5: 같은 split인데 한쪽에만 있고 다른 쪽에는 없는 call_id."""
    manifest_ids = set(call_ids(call_manifest))
    segment_ids = {segment["call_id"] for segment in segment_manifest}
    return {
        "segment_only": segment_ids - manifest_ids,
        "manifest_only": manifest_ids - segment_ids,
    }


def invalid_labels(call_manifest: list[dict]) -> list[dict]:
    """label이 0/1이 아니거나, 디렉터리와 label이 어긋나는 레코드."""
    problems = []
    for record in call_manifest:
        label = record.get("label")
        folder_says_vishing = "/vishing/" in record.get("transcript_filepath", "")
        if label not in VALID_LABELS or bool(label) != folder_says_vishing:
            problems.append(record)
    return problems


def source_overlap_report(
    call_splits: dict[str, list[dict]], fss_mapping: list[dict]
) -> dict:
    """보고 6: 그 통화들이 두 개 이상의 split에 걸쳐 있는 FSS 원본 녹음.

    보고서에 나열할 수 있도록 집계값과 문제가 된 원본 목록을 함께 돌려준다.
    통과/실패를 가리는 검사는 아니다 (계획서 §8-A).
    """
    split_of_call = {}
    for split, records in call_splits.items():
        for call_id in call_ids(records):
            split_of_call[call_id] = split

    splits_of_source = defaultdict(set)
    calls_of_source = defaultdict(list)
    unmapped_rows = []
    for row in fss_mapping:
        call_id = call_id_from_manifest_file(row["manifest_file"])
        if call_id not in split_of_call:
            unmapped_rows.append(row)
            continue
        source = row["fss_source_file"]
        splits_of_source[source].add(split_of_call[call_id])
        calls_of_source[source].append((call_id, split_of_call[call_id]))

    crossing = {
        source: sorted(calls)
        for source, calls in calls_of_source.items()
        if len(splits_of_source[source]) > 1
    }
    vishing_call_ids = {
        call_id
        for records in call_splits.values()
        for record in records
        if record["label"] == 1
        for call_id in [record["call_id"]]
    }
    mapped_call_ids = {call_id_from_manifest_file(row["manifest_file"]) for row in fss_mapping}
    return {
        "mapping_rows": len(fss_mapping),
        "distinct_sources": len(splits_of_source),
        "match_type_counts": Counter(row["match_type"] for row in fss_mapping),
        "vishing_calls_without_mapping": sorted(vishing_call_ids - mapped_call_ids),
        "mapping_rows_without_call": unmapped_rows,
        "sources_crossing_splits": crossing,
    }


def transcript_duplicate_key(call_text: str) -> str:
    """두 통화가 "같은 전사본"인지 판단할 때 쓰는 키를 돌려준다.

    키가 같은 두 통화는 중복으로 보고된다.
    P01 결정: 공백만 무시한다. 같은 음성이라도 ASR이 띄어쓰기를 다르게 내기
    때문이다. 이보다 느슨하게 잡으면 짧은 인사말끼리 묶일 위험이 있다.
    """
    return "".join(call_text.split())


def duplicate_transcripts(call_text_by_id: dict[str, str]) -> dict[str, list[str]]:
    """transcript_duplicate_key가 같은 call_id 묶음 (2건 이상인 것만)."""
    calls_by_key = defaultdict(list)
    for call_id, call_text in call_text_by_id.items():
        calls_by_key[transcript_duplicate_key(call_text)].append(call_id)
    return {key: sorted(ids) for key, ids in calls_by_key.items() if len(ids) > 1}
