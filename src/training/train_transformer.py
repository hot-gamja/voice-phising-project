"""model_text로 RoBERTa 파인튜닝 (계획서 §13-14). Train -> Validation만 쓴다.

실행(스모크):  uv run python -m src.training.train_transformer --max-steps 30
실행(전체):    uv run python -m src.training.train_transformer --lr 2e-5
experiments/roberta_lr{lr}_val_results.md와 roberta_val_scores.json을 쓴다(최적 LR은
이후 scripts/compare_models.py가 고른다). 여기서 Test는 절대 로드하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.model_text import TOKENIZER_NAME, load_model_text_split
from src.evaluation.threshold import TARGET_RECALL, candidate_thresholds, metrics_at, select_threshold

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
MAX_LENGTH = 512
SEED = 42


def to_dataset(rows: list[dict], tokenizer) -> Dataset:
    dataset = Dataset.from_dict({
        "text": [row["model_text"] for row in rows],
        "label": [row["label"] for row in rows],
    })
    # model_text는 이미 길이가 맞춰져 있다. 여기 절단은 안전망일 뿐이다.
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True, remove_columns=["text"],
    )


def vishing_scores(logits: np.ndarray) -> list[float]:
    """클래스 1의 softmax 확률."""
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1]
    return probabilities.tolist()


def make_compute_metrics(labels: list[int]):
    def compute_metrics(eval_prediction):
        scores = vishing_scores(eval_prediction.predictions)
        point = select_threshold(labels, scores, candidates=candidate_thresholds(scores))
        default = metrics_at(labels, scores, 0.5)
        return {
            "precision_at_r90": point.precision, "recall_at_r90": point.recall,
            "f1_at_r90": point.f1, "thr_at_r90": point.threshold, "f1_at_0.5": default.f1,
        }
    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke run: stop after N steps")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    train_rows = load_model_text_split("train")
    val_rows = load_model_text_split("val")
    train_dataset = to_dataset(train_rows, tokenizer)
    val_dataset = to_dataset(val_rows, tokenizer)
    val_labels = [row["label"] for row in val_rows]

    model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=2)
    tag = "smoke" if args.max_steps > 0 else f"lr{args.lr:g}"
    output_dir = Path("models") / f"roberta_{tag}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="no" if args.max_steps > 0 else "epoch",
        save_strategy="no",
        logging_steps=10,
        seed=SEED,
        bf16=torch.backends.mps.is_available(),
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer, compute_metrics=make_compute_metrics(val_labels),
    )

    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started
    steps = train_result.global_step
    sec_per_step = elapsed / max(steps, 1)
    steps_per_epoch = int(np.ceil(len(train_dataset) / (args.batch_size * args.grad_accum)))
    print(f"\n[timing] device={model.device} steps={steps} elapsed={elapsed:.0f}s sec/step={sec_per_step:.2f} "
          f"steps/epoch={steps_per_epoch} est 3 epochs={sec_per_step * steps_per_epoch * 3 / 60:.0f} min")
    if torch.backends.mps.is_available():
        print(f"[memory] mps peak allocated={torch.mps.driver_allocated_memory() / 2**30:.1f} GiB")
    if args.max_steps > 0:
        return

    prediction = trainer.predict(val_dataset)
    scores = vishing_scores(prediction.predictions)
    point = select_threshold(val_labels, scores, candidates=candidate_thresholds(scores))
    default = metrics_at(val_labels, scores, 0.5)
    lines = [f"# RoBERTa (klue/roberta-base) LR={args.lr:g}, epochs={args.epochs:g}, batch={args.batch_size}x{args.grad_accum} — Train → Validation", "",
             f"- sec/step {sec_per_step:.2f}, total {elapsed / 60:.0f} min", "",
             "| point | threshold | precision | recall | F1 |", "|---|---:|---:|---:|---:|",
             f"| default 0.5 | 0.5000 | {default.precision:.3f} | {default.recall:.3f} | {default.f1:.3f} |",
             f"| Recall ≥ {TARGET_RECALL} (§11) | {point.threshold:.4f} | {point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} |",
             "", "## Per-epoch eval log", ""]
    for entry in trainer.state.log_history:
        if "eval_precision_at_r90" in entry:
            lines.append(f"- epoch {entry['epoch']:.0f}: P@R90 {entry['eval_precision_at_r90']:.3f}, R {entry['eval_recall_at_r90']:.3f}, "
                         f"F1 {entry['eval_f1_at_r90']:.3f}, F1@0.5 {entry['eval_f1_at_0.5']:.3f}, loss {entry['eval_loss']:.4f}")
    Path(f"experiments/roberta_{tag}_val_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(f"experiments/roberta_{tag}_val_scores.json").write_text(json.dumps(
        {"model": f"roberta_{tag}", "split": "val", "lr": args.lr, "decision_threshold": point.threshold,
         "scores": {row["call_id"]: s for row, s in zip(val_rows, scores)}}, ensure_ascii=False, indent=1), encoding="utf-8")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "decision_threshold.json").write_text(json.dumps({"decision_threshold": point.threshold}), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
