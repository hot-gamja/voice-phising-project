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

## Token length (klue/roberta-base, plan §5 / §6 분포)

- tokenizer class: BertTokenizer
- special tokens added per call: 2 → text budget 510 (head 128 + tail 382)

| split | class | n | median | mean | max | > budget (truncated) |
|---|---|---:|---:|---:|---:|---:|
| train | non_vishing | 569 | 273 | 353 | 1873 | 113 (20%) |
| train | vishing | 565 | 330 | 825 | 12190 | 215 (38%) |
| val | non_vishing | 71 | 281 | 378 | 1642 | 17 (24%) |
| val | vishing | 70 | 357 | 861 | 9385 | 24 (34%) |
| test | non_vishing | 71 | 225 | 321 | 1303 | 15 (21%) |
| test | vishing | 71 | 331 | 856 | 8219 | 27 (38%) |

## Train 직접 읽기 (20 calls, seed 42, plan §6)

`scripts/dump_train_reading_samples.py`로 뽑은 정상 10 / 피싱 10. 각 class에서 짧은 쪽 5, 긴 쪽 5. 원문은 재배포 금지라 관찰만 기록한다. Validation·Test는 읽지 않았다.

### 정상 (non_vishing) 10건

| call_id | 길이 | 유형 | 메모 |
|---|---:|---|---|
| d02j11s003326 | 67 | 위치 문의 | 4문장. 내용 없음에 가깝다 |
| d01j021s006801 | 128 | 교재 교환 | 짧은 전자상거래 상담 |
| d01j04s005571 | 193 | 배송비 재결제 | **금액·결제·카드 취소 등장** |
| d02j10s003391 | 202 | 통역 부서 문의 | 콜센터 |
| d02j08s004959 | 326 | 구청 민원 연결 | "무엇을 도와드릴까요?" 시작 |
| d04j18s022783 | 853 | 교재 재구매 | 본인 확인(성함) 절차 있음 |
| d04j18s006112 | 884 | 교재 환불 | **아이디·성함 확인, 5만 원 카드 취소** |
| d02j07s007456 | 958 | 수도요금 가구 분할 | 구청 콜센터, 신분증·고지서 언급 |
| d01j020s006316 | 2,316 | 인강 환불 | **계좌번호·예금주·환불 금액·부모 동의**. 금융 표현 가장 많음 |
| d04j18s001448 | 3,735 | 환불 클레임 | 고객이 항의, 관리자 연결 요구. 긴 정상 통화의 전형 |

### 피싱 (vishing) 10건

| call_id | 길이 | 유형 | 메모 |
|---|---:|---|---|
| vishing_7 | 104 | 대출 사기 ARS | **화자 1명, 녹음 안내**. 은행명 + 정부 자금 + 버튼 유도 |
| vishing_596 | 168 | 검찰 사칭 도입부 | "서울중앙지검", "연루된 사건". 통화가 초반에 끊김 |
| vishing_110 | 304 | 검찰 사칭 | 대포통장 압수, 개설 지점·날짜 제시. 피해자가 경찰 확인하겠다고 하자 "검찰로 전화하라" |
| vishing_60 | 347 | 검찰 사칭 | "200여 명 연루", "피의자 입증". FSS 매핑에서 cross-split로 잡힌 원본과 같은 시나리오 |
| vishing_215 | 364 | 검찰 사칭 | vishing_110과 거의 같은 대본(광명시 지점, 압수 목록). **같은 대본의 변형이 여러 call에 존재** |
| vishing_440 | 1,265 | 납치 협박 | 욕설, 협박. 금융 용어 거의 없음. 다른 피싱과 어휘가 완전히 다름 |
| vishing_288 | 1,394 | 검찰 사칭 후반 | 사건번호 육성 확인, 구속영장 협박, 은행 방문 시 행동 지시. 화자 3명 |
| vishing_650 | 1,691 | 대출 사기 후반 | 이미 입금한 피해자에게 추가 360만 원 요구. "코드가 막혔다" |
| vishing_28 | 1,799 | 대출 사기 | "당일 만기 상환 제도", 카드론으로 받아 갚으면 등급 상향. 화자 2명 |
| vishing_358 | 3,596 | 대출 사기 마무리 | 서류 방문, 금리 안내, **"중개업체·산와머니 연락 받지 마라"** |

### 관찰 요약

1. **정상 통화에도 금융 표현이 있다.** 10건 중 3건(d01j04s005571, d04j18s006112, d01j020s006316)에 계좌·카드 취소·환불 금액·본인 확인이 나온다. "금융 단어 = 피싱"으로는 안 되고, 이것이 V1에서 TF-IDF가 틀릴 후보다.
2. **피싱은 크게 세 유형.** 검찰 사칭(5건), 대출 사기(4건), 납치 협박(1건). 앞 둘은 "기관명 + 사건/대출 + 절차 지시" 구조가 반복되고, 납치 협박은 어휘가 전혀 다르다.
3. **같은 대본의 변형이 있다.** vishing_110과 vishing_215는 지점·날짜·압수 목록이 거의 같다. 원본 녹음이 다르더라도 조직이 같은 대본을 쓴 결과다. Train 안에서는 문제가 없지만, Val/Test에 같은 대본이 있으면 실제보다 점수가 높게 나온다. 이건 call_id나 FSS 매핑으로는 잡히지 않는다. **V2 과제로 기록.**
4. **형식 차이 (V2 D-FMT 후보, V1에서는 특징으로 쓰지 않음).**
   - 정상 통화는 콜센터 상담원의 정형 인사("무엇을 도와드릴까요?", "확인 감사합니다")로 시작·끝나는 경우가 많다.
   - 피싱 통화는 도입부 없이 중간부터 시작하거나(vishing_650, vishing_358) 도입부만 있고 끊긴다(vishing_596). 금감원 공개 녹음이 편집된 조각이기 때문으로 보인다.
   - 피싱 쪽은 화자 1명 독백 구간이 길다. vishing_7은 전부, vishing_28·358은 대부분 [1]이 말한다.
   - 이 셋은 내용이 아니라 데이터 출처의 흔적이다. 모델이 이것을 배우면 실제 통화에서는 통하지 않는다.
5. **ASR 오류는 흔하지만 읽는 데 지장은 없다.** "꺾여버렸는데", "복을 약속하는", "산화머니"(산와머니), "두부를 압수"(두 부). 고유명사와 숫자가 특히 흔들린다. 모델 입력도 같은 텍스트라 별도 정정은 하지 않는다.
6. **Head+Tail 관점.** 긴 피싱 5건 모두 앞 128토큰 안에 기관명이나 대출 조건이 나오고, 뒤쪽에 행동 지시(방문·입금·연락 차단)가 온다. 긴 정상 5건은 앞이 용건, 뒤가 마무리 인사다. 앞뒤만 봐도 구분 가능한 편이나, 중간에만 나오는 금액·계좌 지시는 잘린다.

### P01 완료 기준 답

- 한 행 = call manifest의 통화 1건. transcript json 안에 segment(발화) 수십 개가 들어 있다.
- Label 1 = 피싱(금감원 녹음), 0 = 정상(AI Hub 상담 녹음).
- 하나의 Call은 segment manifest에서 여러 행으로 나뉘어 있다. call manifest에서는 1행.
- 누수 경로: (a) segment 단위 분할 — 공식 split은 call 단위라 없음. (b) FSS 원본 분할 — 2건 발견. (c) 같은 대본의 변형 — 검사 방법 없음, V2.
- Transcript 길이: 글자 수 중앙값 정상 540 / 피싱 658, 평균 704 / 1,661. 피싱 쪽 초장문이 평균을 끌어올린다.
- Head+Tail 512로 잘리는 Call: Train 29% (정상 20%, 피싱 38%).

## P02 최소 전처리 결과 (2026-09-27)

- `src/data/preprocess.py`: segment text를 순서대로 공백 하나로 이어 붙이고, 앞뒤·연속 공백만 정리. 숫자·금액·금융 단어·조사 제거 없음 (plan §7). 화자 태그 없음 (plan §5).
- 출력: `data/processed/{train,val,test}.jsonl` = `call_id, label, text`. 1,134 / 141 / 142건, 빈 텍스트 0.
- `src/data/truncate.py`: Head 128 + Tail (510 − 128 = 382) 토큰. 토큰 경계를 글자 위치(offset)로 되돌려 **문자열**로 출력. 학습·서빙·TF-IDF 공용.
- **구분자 결정: 빈칸 `" "`.** "..." 같은 잘림 표시는 피싱이 2배 더 자주 잘리는 이 데이터에서 길이 힌트가 되므로 넣지 않는다.

| 항목 | 정상 | 피싱 |
|---|---:|---:|
| 잘리는 call | 145 / 711 (20%) | 266 / 706 (38%) |
| 잘려 나간 토큰 비율 (전체 토큰 대비) | 17.3% | **59.8%** |

- 잘린 call의 최종 길이(special 포함): 512가 379건, 511이 31건, 510이 1건. 최대 512, 초과 0. 이음새에서 재토큰화 시 1~2 토큰이 합쳐질 수 있어 "정확히 512"가 아니라 "≤ 512, 510 이상"이 실제 보장 범위다. `tests/test_truncate.py`도 이 기준.
- 해석: 피싱 통화 토큰의 60%가 모델에 보이지 않는다. 앞 128과 뒤 382만 보고 판정하는 것이 V1 RoBERTa의 실제 조건이며, README 한계에 적는다. Head+Tail vs Chunk 비교는 V2.
