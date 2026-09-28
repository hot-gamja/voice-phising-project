# RoBERTa (klue/roberta-base) LR=5e-05, epochs=3, batch=16x1 — Train → Validation

- sec/step 1.47, total 5 min

| point | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| default 0.5 | 0.5000 | 1.000 | 0.986 | 0.993 |
| Recall ≥ 0.9 (§11) | 0.0023 | 1.000 | 1.000 | 1.000 |

## Per-epoch eval log

- epoch 1: P@R90 1.000, R 0.986, F1 0.993, F1@0.5 0.986, loss 0.0688
- epoch 2: P@R90 1.000, R 0.986, F1 0.993, F1@0.5 0.993, loss 0.0450
- epoch 3: P@R90 1.000, R 1.000, F1 1.000, F1@0.5 0.993, loss 0.0446
