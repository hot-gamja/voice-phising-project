"""P01: klue/roberta-base tokenizer로 측정한 모든 통화의 토큰 길이.

실행: uv run python -m scripts.token_length_stats
집계 수치만 다룬다(Test 텍스트는 절대 출력하지 않는다). P01 보고서에 추가한다.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from transformers import AutoTokenizer

from src.data.load_korccvi import CALL_SPLITS, load_all_call_splits, load_transcript, segment_texts

MODEL_NAME = "klue/roberta-base"
MAX_LENGTH = 512
HEAD_TOKENS = 128
REPORT_PATH = Path("experiments/p01_audit_report.md")
LABEL_NAME = {0: "non_vishing", 1: "vishing"}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    text_budget = MAX_LENGTH - special_count
    tail_tokens = text_budget - HEAD_TOKENS

    lines = ["", "## Token length (klue/roberta-base, plan §5 / §6 분포)", ""]
    lines.append(f"- tokenizer class: {type(tokenizer).__name__}")
    lines.append(f"- special tokens added per call: {special_count} → text budget {text_budget} "
                 f"(head {HEAD_TOKENS} + tail {tail_tokens})")
    lines += ["", "| split | class | n | median | mean | max | > budget (truncated) |", "|---|---|---:|---:|---:|---:|---:|"]

    for split, records in load_all_call_splits().items():
        lengths_by_label = {0: [], 1: []}
        for record in records:
            texts = segment_texts(load_transcript(record["transcript_filepath"]))
            call_text = " ".join(text.strip() for text in texts if text.strip())
            token_count = len(tokenizer(call_text, add_special_tokens=False)["input_ids"])
            lengths_by_label[record["label"]].append(token_count)
        for label in (0, 1):
            lengths = lengths_by_label[label]
            truncated = sum(length > text_budget for length in lengths)
            lines.append(
                f"| {split} | {LABEL_NAME[label]} | {len(lengths)} | {statistics.median(lengths):.0f} "
                f"| {statistics.mean(lengths):.0f} | {max(lengths)} | {truncated} ({100 * truncated / len(lengths):.0f}%) |"
            )

    report = "\n".join(lines) + "\n"
    print(report)
    with REPORT_PATH.open("a", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"appended to {REPORT_PATH}")


if __name__ == "__main__":
    main()
