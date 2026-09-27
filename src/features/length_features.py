"""D-LEN features (plan §8-B): length-only numbers from the full, untruncated call."""

from __future__ import annotations

from src.data.load_korccvi import load_call_manifest, load_transcript, segment_texts
from src.data.preprocess import clean_text

FEATURE_NAMES = ("segment_count", "char_count", "avg_segment_chars")


def length_features(transcript: dict) -> list[float]:
    """[segment_count, char_count, avg_segment_chars] for one transcript."""
    segments = [clean_text(text) for text in segment_texts(transcript)]
    segments = [segment for segment in segments if segment]
    segment_count = len(segments)
    char_count = sum(len(segment) for segment in segments)
    avg_segment_chars = char_count / segment_count if segment_count else 0.0
    return [float(segment_count), float(char_count), avg_segment_chars]


def length_features_for_split(split: str) -> tuple[list[list[float]], list[int], list[str]]:
    """(X, y, call_ids) for an official split, in manifest order."""
    features, labels, call_ids = [], [], []
    for record in load_call_manifest(split):
        transcript = load_transcript(record["transcript_filepath"])
        features.append(length_features(transcript))
        labels.append(int(record["label"]))
        call_ids.append(record["call_id"])
    return features, labels, call_ids
