"""P09 step 1: Freeze. Record exactly what will be evaluated on Test, once.

Run:  uv run python -m scripts.freeze
Writes experiments/freeze.json with model paths, Validation-chosen thresholds,
file hashes, and the git commit. evaluate_test.py refuses to run without it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FREEZE_PATH = Path("experiments/freeze.json")
FROZEN_MODELS = {
    "dlen": "models/dlen.joblib",
    "tfidf_logistic": "models/tfidf_logistic.joblib",
    "tfidf_svm": "models/tfidf_svm.joblib",
    "roberta": "models/roberta_lr2e-05",
}
FROZEN_CODE = ["src/data/preprocess.py", "src/data/truncate.py", "src/data/model_text.py",
               "src/features/length_features.py", "src/evaluation/threshold.py"]


def sha256_of(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256()
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                digest.update(file_path.name.encode())
                digest.update(file_path.read_bytes())
        return digest.hexdigest()[:16]
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def threshold_of(name: str, path: Path) -> float:
    if path.is_dir():
        return json.loads((path / "decision_threshold.json").read_text())["decision_threshold"]
    import joblib
    return float(joblib.load(path)["decision_threshold"])


def main() -> None:
    if FREEZE_PATH.exists():
        raise SystemExit(f"{FREEZE_PATH} already exists. Freeze happens once; delete it deliberately to redo.")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "working_tree_clean": not dirty,
        "final_model": "tfidf_logistic",
        "models": {name: {"path": path, "sha256_16": sha256_of(Path(path)), "decision_threshold": threshold_of(name, Path(path))}
                   for name, path in FROZEN_MODELS.items()},
        "code": {path: sha256_of(Path(path)) for path in FROZEN_CODE},
        "rule": "Nothing above changes after this point. Test is evaluated once per model (plan §8-A, §12).",
    }
    FREEZE_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
