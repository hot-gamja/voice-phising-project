# 코딩 가이드라인 — voice-phising-project

전역 `~/.claude/CLAUDE.md`를 그대로 따르되, 아래 항목만 이 프로젝트에서 다르게 적용한다.

## 주석과 커밋 메시지 언어 — 한국어

코드 주석, 독스트링, 커밋 메시지는 **한국어**로 작성한다. 전역 규칙("코드 주석은 항상 영어")을 이 프로젝트에서만 덮어쓴다.

이 프로젝트는 학습 자료를 겸하므로 코드와 히스토리의 주 독자가 작성자 자신이다. 다시 읽을 수 없는 기록은 남겨도 쓸모가 없다.

- 바뀌는 것은 설명문뿐이다. 식별자, 문자열 리터럴, 로그·에러 메시지는 영어를 유지한다.
- 커밋 제목의 `type:` 접두어(`feat`/`docs`/`fix`/`chore`)는 영어로 둔다. 사람이 읽는 설명이 아니라 분류용 토큰이다. 제목 본문과 본문 단락만 한국어로 쓴다 — 예: `feat: 출처 컬럼을 담은 V2 표준 행 추가`.
- 전역 규칙의 커밋 메시지 형식은 그대로다 — 괄호 스코프 없음, 작성자 트레일러 없음, 제목 + 선택적 본문에서 끝낸다.
- 이미 영어로 쓰인 과거 커밋은 다시 쓰지 않는다. `v1.0-final` 태그와 푸시된 히스토리를 고치는 비용이 일관성보다 크다.
- 정형화된 doc 주석 스타일은 유지한다 — Python 독스트링 형식 그대로 두고 내용만 한국어로 쓴다.
- 간결하게 쓰는 규칙은 그대로다. 코드를 다시 설명하지 말고, 그 단위가 무엇을 왜 하는지만 적는다.

### 예외: freeze된 파일 5개는 영어 주석을 유지한다

`experiments/freeze.json`의 `code` 목록에 있는 파일은 **주석도 바꾸지 않는다.**

- `src/data/preprocess.py`
- `src/data/truncate.py`
- `src/data/model_text.py`
- `src/features/length_features.py`
- `src/evaluation/threshold.py`

주석 한 글자만 바꿔도 sha256이 달라져 `scripts/verify_freeze.py`가 실패하고, "V1을 수정하지 않았다"는 V2의 전제가 깨진다. 이 파일을 고쳐야 할 이유가 생기면 번역이 아니라 freeze 정책부터 다시 논의한다.

이 파일들에 기능을 덧붙일 때는 수정하지 말고 **새 모듈에서 import해 재사용**한다. `src/data/canonical.py`(`preprocess.py` 재사용)와 `src/evaluation/metrics.py`(`threshold.py` 옆에 신설)가 그 예다.
