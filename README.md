# 통화 문맥 기반 보이스피싱 위험 탐지 — V1 Prototype

한국어 보이스피싱 통화 transcript를 분류하는 ML 실험과, 그 모델을 서빙하는 FastAPI/WebSocket 백엔드 프로토타입입니다.
**ML 기본기 학습 + AI Backend Prototype**이 목적이며, "작지만 믿을 수 있는 실험 + 작지만 제대로 된 Backend"를 원칙으로 했습니다.

- 계획서(확정본): [docs/voice_phishing_project_plan_v1.2.md](docs/voice_phishing_project_plan_v1.2.md) — §0-A에 단계별 실측 결과가 누적되어 있습니다.
- 용어집: [docs/glossary.md](docs/glossary.md)
- 기간: 2026-09-25 ~ 2026-10-05 (P01~P10은 09-28에 완료)

## 한 줄 결론

> 공개 데이터셋 KorCCVi v2에서는 **글자 n-gram TF-IDF + Logistic Regression이 Test 142건을 전부 맞혔고**, fine-tuned RoBERTa도 3건만 틀렸다.
> 그러나 이 점수는 "피싱을 이해했다"가 아니라 **"피싱(금감원 녹음)과 정상(AI Hub 상담 녹음)의 출처를 구분했다"**에 가깝다. 길이만 보는 진단 모델(D-LEN)이 Precision 0.55에 그친 것, RoBERTa가 틀린 3건이 전부 "금융 어휘가 있는 정상 상담"과 "비금융 기관 사칭"이었던 것이 그 근거다.

## 아키텍처

```mermaid
flowchart LR
    subgraph offline[학습 (Mac M5, MPS)]
        A[KorCCVi v2<br/>transcripts] --> B[preprocess.py<br/>segment 결합 + 공백 정리]
        B --> C[truncate.py<br/>Head 128 + Tail 382 토큰<br/>→ model_text 문자열]
        C --> D1[TF-IDF char 2-3gram<br/>+ Logistic / SVM]
        C --> D2[klue/roberta-base<br/>fine-tune]
        A --> D0[D-LEN<br/>길이 3특징 + Logistic]
        D0 & D1 & D2 --> E[Validation에서<br/>threshold 선택 §11<br/>모델 선택 §15]
        E --> F[Freeze → Test 1회]
    end
    subgraph online[서빙 (Docker, CPU)]
        G[Client] -->|POST /predict| H[FastAPI]
        G -->|WS /ws segment 스트림| I[SessionStore<br/>세션별 context 누적]
        I --> H
        H --> J[Predictor<br/>같은 truncate.py<br/>+ tfidf_logistic.joblib]
        J -->|label, model_score,<br/>decision_threshold| G
    end
```

학습과 서빙이 **같은 `truncate_head_tail` 함수**를 호출합니다. 모델이 학습 때 본 텍스트 형태와 서빙 때 받는 형태가 어긋나지 않게 하기 위해서입니다.

## 데이터

**KorCCVi v2** — https://github.com/selfcontrol7/Korean_Voice_Phishing_Detection

| 구분 | 출처 | Calls | 비고 |
|---|---|---:|---|
| Vishing (1) | 금융감독원 "보이스피싱 지킴이" 공개 녹음 | 706 | 원본 녹음 705개(매핑 기준)를 분할·병합 |
| Non-vishing (0) | AI Hub 저음질 전화망 음성인식 데이터 (교육/공공/HR/전자상거래 상담) | 711 | |
| 공식 split | Train 1,134 / Validation 141 / Test 142 | | call 단위, 그대로 사용 |

- 전사는 네이버 CLOVA Speech ASR 출력이며 오인식이 포함되어 있습니다. 따로 정정하지 않았습니다.
- **라이선스 안내.** KorCCVi 저장소의 코드는 MIT이지만, **transcript와 manifest는 제3자 녹음에서 파생된 것으로 재배포가 금지**되어 있습니다(저장소 `DATA.md` §2). 이 프로젝트는 데이터를 포함하지 않으며(`data/`는 gitignore), 각자 원 출처에서 받아야 합니다. `experiments/` 안의 결과 파일에는 call_id와 점수만 있고 transcript 본문은 없습니다.
- 감사(audit) 보고서: [experiments/p01_audit_report.md](experiments/p01_audit_report.md). 공식 split은 call_id 기준 누수가 없었고, 원본 녹음 기준으로는 **cross-split 2건**이 있습니다(V1은 기존 연구와의 비교를 위해 공식 split 유지).

## 실험 결과

### Validation (141건, 피싱 70) — 모델·threshold 선택용

| 모델 | 입력 | threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| D-LEN (길이 3특징) | 전체 통화 | 0.253 | 0.534 | 0.900 | 0.670 |
| TF-IDF char(2,3) + Logistic, C=1 | model_text | 0.413 | 1.000 | 1.000 | 1.000 |
| TF-IDF char(2,3) + LinearSVC, C=0.1 | model_text | −0.140 (margin) | 1.000 | 1.000 | 1.000 |
| klue/roberta-base, LR 2e-5, 3 epoch | model_text | 0.500 | 1.000 | 0.986 | 0.993 |
| (참고) TF-IDF Logistic, 전체 transcript | 전체 통화 | 0.43 | 1.000 | 0.986 | 0.993 |

- 선택 규칙(§15): Recall ≥ 0.90인 threshold에서 Precision 비교 → 동률(피싱 2건 이내) → F1 → CPU latency → 모델 크기. 세 모델이 동률이라 **가장 빠르고 작은 TF-IDF Logistic이 V1 최종 모델**입니다.
- threshold 정책(§11): Validation 점수를 정렬해 이웃한 두 점수의 중간값을 후보로 두고, 확률 점수는 [0.05, 0.95] 안에서만 고릅니다. RoBERTa는 이 제한이 없으면 예외 통화 1건을 잡으려고 threshold 0.007을 골랐습니다.
- RoBERTa LR 2e-5 / 3e-5 / 5e-5는 모두 같은 Validation 결과였고, 가장 보수적이고 loss가 낮은 2e-5를 택했습니다. Mac M5(MPS, bf16)에서 3 epoch에 약 6분입니다.

### Test (142건, 피싱 71) — Freeze 후 딱 한 번

| 모델 | threshold | Precision | Recall | F1 | FP | FN | F1 95% CI (bootstrap) |
|---|---:|---:|---:|---:|---:|---:|---|
| D-LEN | 0.253 | 0.546 | 0.915 | 0.684 | 54 | 6 | 0.608 – 0.754 |
| **TF-IDF Logistic (V1 final)** | 0.413 | **1.000** | **1.000** | **1.000** | 0 | 0 | 1.000 – 1.000 |
| TF-IDF SVM | −0.140 | 1.000 | 1.000 | 1.000 | 0 | 0 | 1.000 – 1.000 |
| RoBERTa 2e-5 | 0.500 | 0.972 | 0.986 | 0.979 | 2 | 1 | 0.953 – 1.000 |

- Freeze 기록: [experiments/freeze.json](experiments/freeze.json) (모델·코드 해시, git 커밋 `d9b81b4`). 평가 스크립트는 Freeze 파일이 없거나 Test 점수 파일이 이미 있으면 실행을 거부합니다.
- Test에서 TF-IDF Logistic의 정상 최고 점수 0.371, 피싱 최저 점수 0.618로 간격 0.247입니다(Validation에서는 0.011).
- 오류 0건의 신뢰구간이 1.000–1.000인 것은 "오류율 0"이 아니라 "142건에서 관측되지 않음"입니다. 실제 오류율이 2%라도 142건에서 0건이 나올 확률은 약 6%입니다.

### Error Analysis (보고 전용, [experiments/p09_test_results.md](experiments/p09_test_results.md))

RoBERTa가 틀린 3건은 TF-IDF가 전부 맞혔습니다.

| call | 정답 | RoBERTa | TF-IDF | 내용 |
|---|---|---:|---:|---|
| d02j09s000285 | 정상 | 0.985 | 0.194 | 구청 콜센터, 건설업 면허용 "은행 금융 거래 확인서" 문의 |
| d02j07s003502 | 정상 | 0.999 | 0.357 | 자동이체 통장 변경, "계좌번호 예금주 본인 맞습니까" |
| vishing_679 | 피싱 | 0.281 | 0.732 | 출입국관리사무소 사칭, 비자 거부 → 명의도용 신고 유도. 송금·계좌 단어 없음 |

RoBERTa는 "금융 문맥 → 피싱"을 배웠고 그 규칙이 이 3건에서 틀렸습니다. TF-IDF가 맞힌 이유는 "무엇을 도와드릴까요", "담당자 연결해 드릴 텐데요" 같은 **콜센터 정형구와 금감원 녹음 특유의 화법을 글자 조합으로 셌기 때문**이지, 피싱을 더 잘 이해해서가 아닙니다. Test Error Analysis를 근거로 모델·전처리·threshold를 바꾸지 않았습니다(§12).

## 실행 방법

### 0. 준비

```bash
# Python 3.12 + uv 0.12
uv sync                      # dev, train 그룹 포함 설치 (.venv 생성)
uv run pytest -q             # 26 tests (데이터·모델 artifact 없으면 일부 skip)
```

### 1. 데이터

```bash
mkdir -p data/raw
git clone --depth 1 https://github.com/selfcontrol7/Korean_Voice_Phishing_Detection data/raw/korccvi
```

`data/raw/korccvi/Multimodal/data/`에 manifest와 transcript가 있어야 합니다. 재배포 금지이므로 `data/`는 git에 올라가지 않습니다.

### 2. 실험 재현 (순서대로)

```bash
uv run python -m scripts.audit_korccvi          # P01 무결성 검사 → experiments/p01_audit_report.md
uv run python -m scripts.token_length_stats     # 토큰 길이·잘림 비율 (보고서에 추가)
uv run python -m scripts.build_processed        # P02 data/processed/{train,val,test}.jsonl
uv run python -m scripts.run_dlen               # P03 D-LEN
uv run python -m scripts.run_tfidf_logistic     # P04 (model_text 캐시 생성 포함)
uv run python -m scripts.run_tfidf_svm          # P05
uv run python -m src.training.train_transformer --max-steps 30   # P06 smoke run
uv run python -m src.training.train_transformer --lr 2e-5        # P07 (약 6분, MPS)
uv run python -m scripts.compare_models         # §15 비교표
uv run python -m scripts.freeze                 # P09-1 Freeze (1회)
uv run python -m scripts.evaluate_test          # P09-2 Test Unlock (1회)
```

학습 산출물은 `models/`(gitignore)에 생깁니다. 서빙에 필요한 것은 `models/tfidf_logistic.joblib`과 `models/tokenizer_klue_roberta_base/`입니다.

### 3. 서버

```bash
uv run uvicorn app.main:app --reload            # http://127.0.0.1:8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
     -d '{"text":"서울중앙지검입니다. 고객님 명의 계좌가 범죄에 연루되어 안전계좌로 이체하셔야 합니다."}'
```

```json
{"label":"PHISHING","model_score":0.808,"decision_threshold":0.4125,"model_version":"v1"}
```

- `model_score`는 보정되지 않은 모델 출력값입니다. **현실 세계의 피싱 확률이 아닙니다.** 그래서 `decision_threshold`를 응답에 같이 실어 "왜 이 점수에서 PHISHING인지"를 클라이언트가 알 수 있게 했습니다.
- 모델은 앱 시작 시 `lifespan`에서 한 번 로드하고, 추론은 `run_in_threadpool`에서 돌아 다른 요청을 막지 않습니다.

### 4. WebSocket 데모 (Mock Transcript Stream)

```bash
uv run python -m scripts.mock_stream            # Validation 통화 3건을 segment 단위로 재생
```

메시지 형식은 [app/schemas.py](app/schemas.py)에 있습니다. Client → `start / segment(seq, text) / end`, Server → `score(model_score, label, decision_threshold, context_segments)` 또는 `error`. 세션별로 segment를 누적한 전체 문맥에 Head+Tail을 적용해 매번 점수를 냅니다. `seq`가 역전되거나 텍스트가 비면 `error`를 돌려줍니다.

재생 로그 예시([experiments/demo_stream_log.txt](experiments/demo_stream_log.txt)):

```text
=== d02j12s000729 (정답: 정상, 12 segments) ===
seq   1 | 0.415 ########            | PHISHING | 뭐 좀 여쭤볼게요. 거기 피 검사도 해주나요?
seq   2 | 0.574 ###########         | PHISHING | 네. 고객님 심혈관 질환에 대해서 저희가 혈액 검사 진행이 되고요.
...
seq  12 | 0.407 ########            | NORMAL   | 보건소 진료실에 먼저 접수해 주시고요.
```

정상 통화가 중간에는 PHISHING으로 흔들리다 전체 문맥에서 NORMAL로 끝납니다. 아래 한계 2번입니다.

### 5. Docker

```bash
docker compose up --build                       # CPU 추론, 이미지 약 1.9GB
```

이미지에는 `models/tfidf_logistic.joblib`과 토크나이저가 함께 들어가며 `HF_HUB_OFFLINE=1`로 기동 시 네트워크에 나가지 않습니다. 이미지를 만들려면 먼저 2번의 학습 스크립트를 돌려 artifact가 있어야 합니다. Linux에서는 pyproject의 마커로 CPU 전용 torch index가 선택됩니다.

## 한계 (Limitations)

1. **출처-라벨 confound.** 피싱은 전부 금감원 녹음, 정상은 전부 AI Hub 상담 녹음입니다. Test 만점은 두 출처의 어휘·말투·ASR 흔적을 구분한 결과일 수 있으며, D-LEN(길이만) 0.55 → TF-IDF 1.00의 향상폭이 "피싱 문맥" 때문인지 "출처 표면 패턴" 때문인지 이 데이터로는 분리할 수 없습니다. 실제 통화에서의 성능은 주장하지 않습니다.
2. **prefix 점수는 검증되지 않았습니다.** 모델은 통화 전체(Head+Tail)로 학습됐고, WebSocket 데모의 초반 점수는 학습 분포 밖입니다. 위 데모 로그처럼 정상 통화가 중간에 PHISHING으로 흔들립니다.
3. **Test는 142건입니다.** 모델 간 작은 차이는 유의하다고 주장하지 않으며, 오류 0건도 오류율 0을 뜻하지 않습니다.
4. **Validation 141건으로 고른 모델·threshold는 노이즈가 있습니다.** Validation에서 TF-IDF의 정상/피싱 간격은 0.011에 불과했습니다. 교차검증은 하지 않았습니다.
5. **Recall 0.90은 V1 데모용 임시 operating point입니다.** 실제 피해 비용이나 업무 근거에서 나온 값이 아닙니다. 실제 서비스 threshold는 FP/FN 비용과 사용자 경고 허용 수준을 기반으로 별도 결정해야 합니다.
6. **Head+Tail 절단.** Train 통화의 29%(피싱 38%, 정상 20%)가 512토큰을 넘어 잘리며, **피싱 토큰의 60%**가 모델에 보이지 않습니다. 잘림 구분자는 길이 힌트를 주지 않도록 빈칸입니다.
7. **원본 녹음 기준 누수 2건.** 공식 split에서 같은 금감원 원본이 Train/Val, Train/Test에 나뉜 경우가 각 1건 있습니다. call_id 기준 누수는 없습니다. 또한 같은 사기 대본의 변형 call이 존재하며 이는 어떤 검사로도 잡히지 않습니다.
8. **In-memory session store.** 단일 인스턴스 검증용입니다. Scale-out 시 Redis 등 외부 State Store가 필요합니다.
9. **ASR 텍스트 기반.** 실제 음성 입력, STT, 실시간성, 거래 차단은 범위 밖입니다.

## V2로 넘긴 것

- **Hard Negative**: 금융 상담(계좌·본인 확인·환불)이 나오는 정상 통화를 추가해 출처 편향을 줄인다. RoBERTa의 FP 2건이 정확히 이 유형이다.
- 비금융 기관 사칭 시나리오(출입국·우체국·택배) 추가 — FN 1건.
- 콜센터 정형구 마스킹 후 TF-IDF가 여전히 만점인지 확인(출처 표면 패턴 진단).
- source-grouped 재분할, 같은 대본 변형 탐지.
- KoELECTRA / KF-DeBERTa 비교, Head+Tail vs Chunk 비교, prefix 학습(V3).

## 저장소 구조

```text
app/                 FastAPI (main.py, schemas.py, service.py, session.py)
src/data/            load_korccvi, split_check, preprocess, truncate, model_text
src/features/        length_features (D-LEN)
src/evaluation/      threshold (§11 정책, PR 표, 선택 규칙)
src/training/        train_transformer (HF Trainer, MPS)
src/inference/       predictor (서빙용, 학습과 같은 truncate 사용)
scripts/             단계별 실행 스크립트 (audit → build → run_* → freeze → evaluate_test → mock_stream)
tests/               split 검사 5, truncate 4, predictor 4, REST 4, WS 5 (총 26)
experiments/         결과 md/json, freeze.json, demo_calls.json (transcript 본문 없음)
docs/                계획서 v1.0 → v1.2, 용어집
models/              (gitignore) 학습 산출물, tokenizer
data/                (gitignore) KorCCVi 원본과 processed
```

## 참고

- KorCCVi v2 — selfcontrol7/Korean_Voice_Phishing_Detection
- KLUE RoBERTa Base — https://huggingface.co/klue/roberta-base
- Hypothesis-only baseline 발상 — Gururangan et al., 2018 (D-LEN의 근거)
- 그 외 선행연구 목록은 계획서 §41
