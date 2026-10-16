# voice-phising-project

전역 `~/.claude/CLAUDE.md`를 따른다. 아래는 이 프로젝트에서만 다르게 적용하는 것.

## 1. 주석·커밋 메시지는 한국어

전역 규칙("코드 주석은 항상 영어")을 이 프로젝트에서만 덮어쓴다. 이 리포는 학습 자료를 겸하고, 코드와 히스토리의 주 독자가 작성자 자신이다.

| 대상 | 언어 |
|---|---|
| 코드 주석, 독스트링 | **한국어** |
| 커밋 메시지 제목 본문·본문 단락 | **한국어** |
| 커밋 제목 `type:` 접두어 (`feat`/`fix`/`docs`/`chore`) | 영어 — 도구가 파싱하는 토큰 |
| 식별자, 문자열 리터럴, 로그·에러 메시지 | 영어 |

문자열처럼 생겼지만 **런타임에 사용자에게 나가는 것은 번역하지 않는다**: pydantic `Field(description=...)`(OpenAPI 스키마), argparse `help=`, pytest `skipif(reason=)`, 보고서 스크립트가 f-string으로 만드는 마크다운.

커밋 메시지 형식은 전역 규칙 그대로 — 괄호 스코프 없음, 작성자 트레일러 없음. 이미 영어로 쓰인 과거 커밋은 다시 쓰지 않는다.

## 2. freeze된 파일 5개는 수정 금지

`experiments/freeze.json`의 `code` 목록:

```
src/data/preprocess.py
src/data/truncate.py
src/data/model_text.py
src/features/length_features.py
src/evaluation/threshold.py
```

**주석 한 글자도 바꾸지 않는다.** 바꾸면 sha256이 달라져 `scripts/verify_freeze.py`가 실패하고, "V1을 수정하지 않았다"는 V2의 전제가 깨진다.

기능을 덧붙일 때는 **수정하지 말고 새 모듈에서 import한다.** 선례:

| 새 모듈 | 재사용 대상 |
|---|---|
| `src/data/canonical.py` | `preprocess.call_text_from_transcript` |
| `src/evaluation/metrics.py` | `threshold.py` 옆에 신설 |

## 3. 작업 전후 확인

```bash
uv run python -m scripts.verify_freeze   # 9/9 match, exit 0
uv run pytest -q                         # 26 passed
```

V1 artifact(`models/`)는 gitignore라 덮어써도 git이 감지하지 못하고 복구 경로도 없다. 학습 스크립트를 재실행하기 전에 출력 경로를 먼저 확인한다.

## 4. 현재 단계 — V2 일반화 검증

V2는 기능 추가가 아니라 **V1의 Test F1 1.000이 쉬운 단서(출처·대본) 때문이 아닌지 검증**하는 단계다. 진행 상황은 `git log --oneline v1.0-final..HEAD`.

계획서는 `docs/`에 있고 `docs/`는 gitignore다 — 로컬에만 존재한다.
