# D-LEN (length-only) — Train → Validation

- features: segment_count, char_count, avg_segment_chars (full call, before truncation)
- model: StandardScaler + LogisticRegression, train n=1134, val n=141 (vishing 70)
- standardized coefficients: segment_count=+2.48, char_count=-1.13, avg_segment_chars=+1.10

## Operating points

| point | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| default 0.5 | 0.50 | 0.697 | 0.657 | 0.676 |
| **Recall ≥ 0.90 (§11)** | 0.25 | **0.534** | 0.900 | 0.670 |

## PR trade-off (Validation)

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 0.496 | 1.000 | 0.664 |
| 0.10 | 0.493 | 0.986 | 0.657 |
| 0.15 | 0.496 | 0.986 | 0.660 |
| 0.20 | 0.516 | 0.943 | 0.667 |
| 0.25 | 0.534 | 0.900 | 0.670 |
| 0.30 | 0.546 | 0.843 | 0.663 |
| 0.35 | 0.582 | 0.814 | 0.679 |
| 0.40 | 0.624 | 0.757 | 0.684 |
| 0.45 | 0.667 | 0.714 | 0.690 |
| 0.50 | 0.697 | 0.657 | 0.676 |
| 0.55 | 0.724 | 0.600 | 0.656 |
| 0.60 | 0.792 | 0.543 | 0.644 |
| 0.65 | 0.860 | 0.529 | 0.655 |
| 0.70 | 0.900 | 0.514 | 0.655 |
| 0.75 | 0.935 | 0.414 | 0.574 |
| 0.80 | 1.000 | 0.257 | 0.409 |
| 0.85 | 1.000 | 0.229 | 0.372 |
| 0.90 | 1.000 | 0.100 | 0.182 |
| 0.95 | 1.000 | 0.057 | 0.108 |
