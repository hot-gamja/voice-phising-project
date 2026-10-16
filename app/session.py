"""WebSocket 모의 스트림용 메모리 세션 저장소 (계획서 §20).

V1 한계: 단일 프로세스 전용. 여러 프로세스로 늘리려면 외부 저장소(Redis)가 필요하다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    segments: list[str] = field(default_factory=list)
    last_seq: int = 0

    def append(self, seq: int, text: str) -> None:
        """다음 구간을 받는다. 순서가 어긋나거나 중복된 seq는 거부한다 (계획서 §19)."""
        if seq <= self.last_seq:
            raise ValueError(f"seq {seq} is not after {self.last_seq}")
        self.segments.append(text)
        self.last_seq = seq

    def context_text(self) -> str:
        """지금까지 누적된 전사본. Head+Tail 절단은 이후 Predictor가 적용한다."""
        return " ".join(self.segments)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def start(self, session_id: str) -> Session:
        session = Session(session_id)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def end(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)
