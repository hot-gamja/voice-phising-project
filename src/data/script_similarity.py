"""대본 간 유사도 측정. 임계값을 정하기 위한 단계이고, 묶지는 않는다.

왜 측정부터 하는가. 커밋 5의 완전 일치는 1,417건에서 2쌍만 잡았다. 그런데 P01은
사람이 읽어서 vishing_215 ≈ vishing_110 같은 "거의 같은 대본"을 더 찾아냈다. 즉
글자 단위 일치로는 안 보이는 사각지대가 있다. 그 사각지대를 클러스터로 묶으려면
"유사도 몇 이상을 같은 대본으로 볼지" 임계값이 필요한데, 한국어 통화는 조사와
전화 상투어("여보세요", "네 알겠습니다")를 공유해서 **무관한 통화끼리도 유사도가
0이 아니다.** 그래서 0.9 같은 숫자를 근거 없이 박으면 안 되고, 분포를 먼저 본다.

묶는 일(connected components)은 다음 단계에서 이 측정 결과를 받아 한다.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.split_check import transcript_duplicate_key

# V1 분류기와 같은 벡터화 설정. 프로젝트에 텍스트 표현이 두 벌 생기지 않게 맞춘다.
VECTORIZER_SETTINGS = {
    "analyzer": "char_wb",
    "ngram_range": (2, 3),
    "min_df": 2,
    "sublinear_tf": True,
}


def similarity_matrix(call_text_by_id: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """(call_id 순서, 쌍별 코사인 유사도 행렬)을 돌려준다.

    TfidfVectorizer는 각 행을 L2 정규화하므로 행끼리의 내적이 곧 코사인 유사도다.
    길이가 나눠지기 때문에, 통화 길이가 달라도 같은 대본이면 높게 나온다.
    자기 자신과의 유사도(대각선)는 1.0이라 의미가 없으므로 0으로 지운다.
    """
    call_ids = sorted(call_text_by_id)
    normalized = [transcript_duplicate_key(call_text_by_id[call_id]) for call_id in call_ids]
    matrix = TfidfVectorizer(**VECTORIZER_SETTINGS).fit_transform(normalized)
    similarity = (matrix @ matrix.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    return call_ids, similarity


def top_pairs(call_ids: list[str], similarity: np.ndarray, count: int = 20) -> list[tuple[str, str, float]]:
    """유사도가 가장 높은 (call_id, call_id, 유사도) 쌍들, 내림차순.

    행렬이 대칭이므로 위쪽 삼각만 보고 같은 쌍을 두 번 세지 않는다.
    """
    rows, columns = np.triu_indices_from(similarity, k=1)
    scores = similarity[rows, columns]
    ranked = np.argsort(scores)[::-1][:count]
    return [(call_ids[rows[index]], call_ids[columns[index]], float(scores[index])) for index in ranked]


def similarity_of(call_ids: list[str], similarity: np.ndarray, left: str, right: str) -> float:
    """특정 두 통화의 유사도. 사람이 찾아둔 쌍을 검산할 때 쓴다."""
    return float(similarity[call_ids.index(left), call_ids.index(right)])


def nearest_neighbor_scores(similarity: np.ndarray) -> np.ndarray:
    """통화별로 "가장 비슷한 다른 통화"와의 유사도.

    이 분포가 임계값 선택의 근거다. 대부분의 통화는 짝이 없으므로 이 값들이
    몰려 있는 구간이 곧 "무관한 통화끼리의 유사도"이고, 임계값은 그 위에 둬야
    한다.
    """
    return similarity.max(axis=1)


def distribution_summary(scores: np.ndarray, percentiles: tuple[float, ...] = (50, 90, 99, 99.9)) -> dict:
    """분포 요약: 개수, 평균, 지정한 분위수, 최대값."""
    return {
        "count": int(scores.size),
        "mean": float(scores.mean()),
        "percentiles": {str(point): float(np.percentile(scores, point)) for point in percentiles},
        "max": float(scores.max()),
    }
