# 대본 유사도 측정 (2026-10-16)

`src/data/script_similarity.py`로 KorCCVi 1,417건 전체의 쌍별 코사인 유사도를 측정했다.
목적은 묶는 것이 아니라 **connected components에 쓸 임계값의 근거를 만드는 것**이다.

설정: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3), min_df=2, sublinear_tf=True)`
— V1 분류기와 같다. 텍스트는 `transcript_duplicate_key`로 공백만 정규화(커밋 5와 동일).
행렬 1,417 × 1,417 (16.1 MB), 쌍 1,003,236개.

## 지표 자체는 검증됐다

커밋 5가 완전 일치로 찾은 2쌍이 정확히 1.0000으로 나온다. 정답이 있는 검산이다.

| 쌍 | 유사도 |
|---|---:|
| `vishing_135` ~ `vishing_200` | 1.0000 |
| `vishing_554` ~ `vishing_555` | 1.0000 |

상위 12쌍은 전부 0.966 이상이고 **전부 피싱끼리**다. 0.96 위에 뚜렷한 덩어리가 있다.

## 통화별 최근접 이웃 유사도 분포

| 통계 | 값 |
|---|---:|
| 평균 | 0.2443 |
| 50 분위 | 0.1803 |
| 90 분위 | 0.4459 |
| 99 분위 | 0.9729 |
| 최대 | 1.0000 |

중앙값 0.18이 "무관한 통화끼리도 이 정도는 닮는다"의 기준선이다. 한국어 통화가
조사와 전화 상투어를 공유하기 때문이고, 그래서 낮은 임계값은 쓸 수 없다.

## 발견 1 — split을 넘나드는 near-duplicate가 실제로 있다

커밋 5의 완전 일치 기준으로는 split을 넘는 클러스터가 **0건**이었다. 유사도로 보면 나온다.

| 임계값 | 전체 쌍 | train~test | train~val | val~test | 영향받는 test 통화 |
|---:|---:|---:|---:|---:|---:|
| 0.95 | 13 | 2 | 1 | 0 | 2 |
| 0.90 | 19 | 2 | 1 | 0 | 2 |
| 0.80 | 26 | 3 | 1 | 1 | 4 |
| 0.70 | 34 | 3 | 2 | 1 | 4 |
| 0.60 | 46 | 5 | 2 | 2 | 7 |
| 0.50 | 70 | 7 | 5 | 3 | 9 |
| 0.30 | 232 | 33 | 30 | 5 | 30 |

가장 심한 쌍:

| 유사도 | 쌍 |
|---:|---|
| 0.9729 | `vishing_60`(train) ~ `vishing_63`(**test**) |
| 0.9662 | `vishing_58`(**test**) ~ `vishing_61`(train) |
| 0.8914 | `vishing_256`(**test**) ~ `vishing_321`(val) |
| 0.8771 | `vishing_262`(train) ~ `vishing_422`(**test**) |

즉 **V1의 Test 142건에는 Train 통화와 0.97로 겹치는 통화가 들어 있었다.** Test F1 1.000의
일부는 일반화가 아니라 사본을 맞춘 결과다. 이것이 V2가 group-aware split을 만드는 직접
근거다. P01이 `vishing_60`을 "FSS 매핑에서 cross-split로 잡힌 원본과 같은 시나리오"로
적어둔 것과도 일치한다.

## 발견 2 — 이 지표는 P01이 사람 눈으로 찾은 쌍을 잡지 못한다

P01 보고서가 지목한 `vishing_215` ≈ `vishing_110`("광명시 지점, 압수 목록이 거의 같다")의
유사도는 **0.1091**이다. 중앙값 0.1803보다도 낮고, `vishing_215` 입장에서 1,416건 중 **22위**다.
`vishing_215`의 최근접 이웃은 `vishing_216`(0.2464)이고 `vishing_110`이 아니다.

원인은 P01의 서술에 이미 있다 — 두 통화는 **대본의 일부**(지점·날짜·압수 목록)를 공유하고,
녹음·길이·피해자 반응은 다르다(304 vs 364 토큰). 통화 전체를 한 벡터로 비교하면 공유된
부분이 나머지에 희석된다.

이 쌍을 잡으려고 임계값을 0.1091까지 내리면:

- 쌍 **24,513개** (전체의 2.4%)
- 관련 통화 **1,298 / 1,417건 (92%)**

connected components는 전이되므로 이 상태로 묶으면 거의 전부가 거대 클러스터 하나가 되고
group split 자체가 불가능해진다. **어떤 임계값으로도 이 쌍을 잡을 수 없다.**

## 결론: 6b에 넘기는 것

1. **임계값 후보는 0.95**다. 쌍 13개로 작고, 전부 눈으로 확인 가능하며, train~test 2쌍을
   잡는다. 0.80까지 내리면 train~test 3쌍·test 4건으로 늘고 여전히 관리 가능하다.
   0.30 아래는 쓰지 않는다.
2. **6b는 최대 클러스터 크기를 반드시 보고해야 한다.** 전이 때문에 임계값을 조금 내려도
   연쇄 병합이 일어난다.
3. **부분 일치 탐지는 이 방법으로 해결되지 않는다.** 통화 전체 코사인은 "같은 대본의 변형"을
   놓친다. 별도 방법(희귀 n-gram 공유, 구간 단위 비교)이 필요하고, 이번 범위가 아니다.
   **따라서 6b가 끝나도 "대본 누수를 다 막았다"고 말할 수 없다.** 막은 것은 통화 전체가
   거의 같은 경우뿐이다.

## 재현

```python
from src.data.canonical import load_canonical_split
from src.data.script_similarity import similarity_matrix, top_pairs, similarity_of

rows = [r for s in ('train','val','test') for r in load_canonical_split(s)]
call_ids, similarity = similarity_matrix({r['call_id']: r['text'] for r in rows})
top_pairs(call_ids, similarity, 12)
similarity_of(call_ids, similarity, 'vishing_215', 'vishing_110')   # 0.1091
```
