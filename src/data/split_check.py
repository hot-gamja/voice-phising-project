"""Integrity checks for the official KorCCVi split (plan §8-A, §6 품질).

Checks 1-5 are pass/fail. Source overlap (6) and duplicates are reports only:
the official split is kept as-is in V1.
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
    """Checks 1-3: call_ids shared between any two splits. Every value must be empty."""
    id_sets = {split: set(call_ids(records)) for split, records in call_splits.items()}
    overlaps = {}
    for split_a, split_b in combinations(id_sets, 2):
        overlaps[f"{split_a}∩{split_b}"] = id_sets[split_a] & id_sets[split_b]
    return overlaps


def duplicate_call_ids_within_split(call_manifest: list[dict]) -> list[str]:
    """call_ids that appear more than once inside one manifest."""
    counts = Counter(call_ids(call_manifest))
    return sorted(call_id for call_id, count in counts.items() if count > 1)


def total_calls(call_splits: dict[str, list[dict]]) -> int:
    """Check 4: sum of the three splits. Must equal EXPECTED_TOTAL_CALLS."""
    return sum(len(records) for records in call_splits.values())


def segment_call_id_mismatch(call_manifest: list[dict], segment_manifest: list[dict]) -> dict[str, set[str]]:
    """Check 5: call_ids present on one side but not the other, for the same split."""
    manifest_ids = set(call_ids(call_manifest))
    segment_ids = {segment["call_id"] for segment in segment_manifest}
    return {
        "segment_only": segment_ids - manifest_ids,
        "manifest_only": manifest_ids - segment_ids,
    }


def invalid_labels(call_manifest: list[dict]) -> list[dict]:
    """Records whose label is not 0/1, or whose folder disagrees with the label."""
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
    """Report 6: FSS source recordings whose calls land in more than one split.

    Returns counts plus the offending sources so the report can list them.
    Not a pass/fail check (plan §8-A).
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
    """Return the key used to decide whether two calls are "the same transcript".

    Two calls whose keys are equal are reported as duplicates.
    Decision (P01): ignore whitespace only. ASR output varies in spacing for
    the same audio; anything looser risks grouping short greetings.
    """
    return "".join(call_text.split())


def duplicate_transcripts(call_text_by_id: dict[str, str]) -> dict[str, list[str]]:
    """Groups of call_ids whose transcript_duplicate_key is identical (size >= 2)."""
    calls_by_key = defaultdict(list)
    for call_id, call_text in call_text_by_id.items():
        calls_by_key[transcript_duplicate_key(call_text)].append(call_id)
    return {key: sorted(ids) for key, ids in calls_by_key.items() if len(ids) > 1}
