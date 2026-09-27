"""Plan §8-A checks 1-5 as pass/fail tests on the real official split."""

import pytest

from src.data.load_korccvi import (
    CALL_SPLITS,
    DATA_ROOT,
    load_all_call_splits,
    load_segment_manifest,
)
from src.data.split_check import (
    EXPECTED_TOTAL_CALLS,
    call_id_overlaps,
    duplicate_call_ids_within_split,
    invalid_labels,
    segment_call_id_mismatch,
    total_calls,
)

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="KorCCVi data not present")


@pytest.fixture(scope="module")
def call_splits():
    return load_all_call_splits()


def test_call_ids_disjoint_between_splits(call_splits):
    for pair, shared in call_id_overlaps(call_splits).items():
        assert not shared, f"{pair} shares {len(shared)} call_ids"


def test_no_duplicate_call_id_within_split(call_splits):
    for split, records in call_splits.items():
        assert duplicate_call_ids_within_split(records) == [], split


def test_total_calls(call_splits):
    assert total_calls(call_splits) == EXPECTED_TOTAL_CALLS


@pytest.mark.parametrize("split", CALL_SPLITS)
def test_segment_manifest_matches_call_manifest(call_splits, split):
    mismatch = segment_call_id_mismatch(call_splits[split], load_segment_manifest(split))
    assert not mismatch["segment_only"], split
    assert not mismatch["manifest_only"], split


@pytest.mark.parametrize("split", CALL_SPLITS)
def test_labels_valid_and_match_folder(call_splits, split):
    assert invalid_labels(call_splits[split]) == []
