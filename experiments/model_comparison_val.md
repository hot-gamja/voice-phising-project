# Model comparison — Validation (n=141, vishing 70)

Rule (§15): precision at recall ≥ 0.9; tie if precision within 2 calls (≈0.031); then F1.

| model | threshold | precision | recall | F1 | note |
|---|---:|---:|---:|---:|---|
| dlen | +0.250 | 0.534 | 0.900 | 0.670 |  |
| tfidf_logistic | +0.418 | 1.000 | 1.000 | 1.000 | **selected** |
| tfidf_svm | -0.126 | 1.000 | 1.000 | 1.000 | tied |
