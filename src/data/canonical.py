"""V2 표준 행: V1의 {call_id, label, text}에 통화의 출처를 더한 형태.

V1의 data/processed/*.jsonl에는 출처 필드가 없어서 데이터셋별 지표를 아예
계산할 수 없다. 일반화 연구에서는 치명적인 결함이라, 이 행들이 그 필드를 더한다.

별도 모듈·별도 출력 경로로 만든 이유: src/data/preprocess.py는
experiments/freeze.json으로 봉인돼 바이트 단위로 유지해야 한다. 그 함수들은
가져다 쓰기만 하고 고치지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.load_korccvi import load_call_manifest, load_transcript
from src.data.preprocess import call_text_from_transcript

CANONICAL_DIR = Path("data/processed/v2")

# 이 모듈이 읽는 데이터셋. 외부 데이터셋은 나중에 자기 빌더를 따로 갖는다.
DATASET = "korccvi_v2"

# KorCCVi가 합쳐온 두 원 녹음 출처. 클래스마다 하나씩이다 (상류 DATA.md §2.1, §2.2).
# AI Hub 데이터셋 번호(dataSetSn)는 식별자에 넣지 않는다. 상류는 데이터셋 이름
#「저품질 전화망 음성인식 데이터」만 적고 번호는 적지 않았고, 우리 계획서도 "#571"을
# 도메인 서술 일치에 근거한 추정이라고 명시했다.
SOURCE_FSS = "fss_voice_phishing_guard"
SOURCE_AIHUB = "aihub_lowquality_telephony"


def source_of(record: dict) -> str:
    """KorCCVi manifest 한 행에서 그 통화의 원 녹음 출처를 돌려준다.

    라벨에서 유도하는 것은 의도된 설계다. KorCCVi 안에서 출처와 라벨은 구성상
    완전히 일치한다 — 피싱은 전부 FSS, 정상은 전부 AI Hub (DATA.md §4) — 그래서
    이 컬럼은 여기서는 추가 정보를 담지 않는다. 그 일치 자체가 V2가 측정하려는
    confound이고, 명시해 두는 덕분에 세 번째 출처가 들어오는 순간 출처별 지표를
    표현할 수 있게 된다. 다른 데이터셋은 자기 매핑을 따로 둬야 한다.
    """
    return SOURCE_FSS if record["label"] == 1 else SOURCE_AIHUB


def build_canonical_split(split: str) -> list[dict]:
    """KorCCVi split 하나의 [{call_id, label, text, dataset, source}], manifest 순서."""
    rows = []
    for record in load_call_manifest(split):
        transcript = load_transcript(record["transcript_filepath"])
        rows.append({
            "call_id": record["call_id"],
            "label": record["label"],
            "text": call_text_from_transcript(transcript),
            "dataset": DATASET,
            "source": source_of(record),
        })
    return rows


def write_canonical_split(split: str) -> Path:
    """data/processed/v2/{split}.jsonl을 쓰고 그 경로를 돌려준다."""
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CANONICAL_DIR / f"{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in build_canonical_split(split):
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def load_canonical_split(split: str) -> list[dict]:
    """data/processed/v2/{split}.jsonl을 다시 읽어 들인다."""
    with (CANONICAL_DIR / f"{split}.jsonl").open(encoding="utf-8") as canonical_file:
        return [json.loads(line) for line in canonical_file if line.strip()]
