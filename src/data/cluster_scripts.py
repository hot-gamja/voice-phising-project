"""대본 클러스터: 같은 대본에서 나온 통화를 한 그룹으로 묶는다.

왜 필요한가. V1의 공식 split은 call_id가 겹치지 않는지만 확인했다. 그런데
같은 대본을 다시 읽은 통화는 call_id가 다르므로 그 검사를 통과하면서 Train과
Test에 동시에 들어갈 수 있다. 그러면 모델은 일반화가 아니라 암기로 점수를
낸다. group-aware split(커밋 8)은 클러스터를 단위로 나눠야 하고, 그 클러스터
id를 만드는 게 이 모듈이다.

이번 단계는 공백만 무시한 완전 일치까지만 다룬다. 표현이 조금씩 다른 유사
대본(P01이 사람 눈으로 발견한 vishing_215 ≈ vishing_110 같은 경우)은 다음
단계에서 추가한다.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from src.data.split_check import transcript_duplicate_key

# 클러스터 id로 쓸 sha256 앞자리 길이. 1417건 규모에서 충돌 확률은 무시 가능하고,
# 로그와 표에서 읽을 수 있을 만큼 짧다.
CLUSTER_ID_LENGTH = 12


def script_cluster_id(call_text: str) -> str:
    """대본 내용으로 결정되는 클러스터 id.

    내용에서 직접 유도하므로 행 순서·split 구성·실행 시점과 무관하게 같은
    대본은 항상 같은 id를 받는다. 일련번호(0, 1, 2...)를 쓰면 입력 순서가
    바뀔 때 id가 전부 흔들려 과거 기록과 대조할 수 없다.
    """
    return hashlib.sha256(transcript_duplicate_key(call_text).encode()).hexdigest()[:CLUSTER_ID_LENGTH]


def cluster_id_by_call(call_text_by_id: dict[str, str]) -> dict[str, str]:
    """{call_id: script_cluster_id}. 혼자인 통화도 자기만의 클러스터를 받는다.

    혼자인 통화를 빼지 않는 이유: split을 나눌 때 모든 통화가 그룹 하나에
    속해 있어야 "그룹 단위로 나눈다"는 규칙을 예외 없이 적용할 수 있다.
    """
    return {call_id: script_cluster_id(text) for call_id, text in call_text_by_id.items()}


def clusters(call_text_by_id: dict[str, str]) -> dict[str, list[str]]:
    """{script_cluster_id: [call_id, ...]} 전체. call_id는 정렬해서 담는다."""
    calls_by_cluster = defaultdict(list)
    for call_id, cluster in cluster_id_by_call(call_text_by_id).items():
        calls_by_cluster[cluster].append(call_id)
    return {cluster: sorted(call_ids) for cluster, call_ids in calls_by_cluster.items()}


def cluster_report(call_text_by_id: dict[str, str]) -> dict:
    """클러스터 요약: 전체 통화 수, 클러스터 수, 2건 이상인 클러스터."""
    grouped = clusters(call_text_by_id)
    shared = {cluster: call_ids for cluster, call_ids in grouped.items() if len(call_ids) > 1}
    return {
        "calls": len(call_text_by_id),
        "clusters": len(grouped),
        "clusters_with_multiple_calls": len(shared),
        "calls_in_shared_clusters": sum(len(call_ids) for call_ids in shared.values()),
        "shared_clusters": shared,
    }


def clusters_crossing_splits(text_by_call_by_split: dict[str, dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    """split을 넘나드는 클러스터: {cluster_id: {split: [call_id, ...]}}.

    비어 있으면 완전 일치 수준에서는 split 간 누수가 없다는 뜻이다. 한 건이라도
    있으면 그 클러스터는 Train과 Test 양쪽에 같은 대본을 넣고 있다.
    """
    splits_of_cluster = defaultdict(lambda: defaultdict(list))
    for split, call_text_by_id in text_by_call_by_split.items():
        for call_id, cluster in cluster_id_by_call(call_text_by_id).items():
            splits_of_cluster[cluster][split].append(call_id)
    return {
        cluster: {split: sorted(call_ids) for split, call_ids in by_split.items()}
        for cluster, by_split in splits_of_cluster.items()
        if len(by_split) > 1
    }
