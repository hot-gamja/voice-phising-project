# Model comparison — Validation (n=141, vishing 70)

Rule (§15): precision at recall ≥ 0.9; tie if precision within 2 calls (≈0.031); then F1.

| model | threshold | precision | recall | F1 | note |
|---|---:|---:|---:|---:|---|
| dlen | +0.253 | 0.534 | 0.900 | 0.670 |  |
| tfidf_logistic | +0.413 | 1.000 | 1.000 | 1.000 | **selected** |
| tfidf_svm | -0.140 | 1.000 | 1.000 | 1.000 | tied |
| roberta | +0.500 | 1.000 | 0.986 | 0.993 | tied |
