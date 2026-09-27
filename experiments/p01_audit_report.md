# P01 KorCCVi v2 Audit Report

## Split integrity (plan §8-A checks 1-5)

- call_id overlap train∩val: 0
- call_id overlap train∩test: 0
- call_id overlap val∩test: 0
- duplicate call_id within train: []
- duplicate call_id within val: []
- duplicate call_id within test: []
- total calls: 1417 (expected 1417)
- train: segment-only call_ids=0, manifest-only call_ids=0, segments=31443
- val: segment-only call_ids=0, manifest-only call_ids=0, segments=4061
- test: segment-only call_ids=0, manifest-only call_ids=0, segments=3925
- invalid labels in train: 0
- invalid labels in val: 0
- invalid labels in test: 0

## Counts

| split | non_vishing | vishing | total |
|---|---:|---:|---:|
| train | 569 | 565 | 1134 |
| val | 71 | 70 | 141 |
| test | 71 | 71 | 142 |

## Transcript quality and length (aggregate)

- empty transcripts: 0 []
- ASR version strings: 1 distinct
  - 1417x `ncp_v2_v2.3.8-22e1a5c-20241203_240905-28192935-dirty_v4.2.12_ko_firedepartment_20240624_`
- every segment has `speaker` field: {True: 1417}
- joined segments == top-level `text` (whitespace-insensitive): {True: 1417}
- duplicate transcript groups (whitespace-insensitive): 2
  - [('vishing_554', 'train'), ('vishing_555', 'train')]
  - [('vishing_135', 'train'), ('vishing_200', 'train')]

### Character length of joined transcript

- train non_vishing: n=569 min=25 median=540 mean=704 max=3735
- train vishing: n=565 min=56 median=658 mean=1661 max=24825
- val non_vishing: n=71 min=102 median=563 mean=750 max=3259
- val vishing: n=70 min=110 median=706 mean=1737 max=19114
- test non_vishing: n=71 min=77 median=432 mean=642 max=2656
- test vishing: n=71 min=89 median=649 mean=1722 max=16571

### Segments per call

- train non_vishing: n=569 min=1 median=14 mean=17 max=89
- train vishing: n=565 min=1 median=14 mean=38 max=605
- val non_vishing: n=71 min=3 median=15 mean=18 max=90
- val vishing: n=70 min=3 median=18 mean=40 max=452
- test non_vishing: n=71 min=2 median=11 mean=16 max=73
- test vishing: n=71 min=2 median=14 mean=39 max=357

## FSS source overlap (plan §8-A report 6)

- mapping rows: 712
- distinct source recordings: 705
- match_type: {'exact': 656, 'offset_corrected': 38, 'concat_part': 6, 'split_part': 9, 'unmatched_local': 1, 'unmatched_backup': 2}
- vishing calls without mapping row: []
- mapping rows without a call: ['', '']
- **sources whose calls cross splits: 2**
  - `200여명이 연루된 사건입니다.mp3` → [('vishing_395', 'train'), ('vishing_537', 'val'), ('vishing_543', 'train')]
  - `[사기 시나리오 3단계] ATM계좌 이체한도 초과1.mp3` → [('vishing_522', 'test'), ('vishing_665', 'train')]
