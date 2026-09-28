"""In-memory session store for the WebSocket mock stream (plan §20).

V1 limitation: single process only. Scale-out needs an external store (Redis).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    segments: list[str] = field(default_factory=list)
    last_seq: int = 0

    def append(self, seq: int, text: str) -> None:
        """Accept the next segment; reject out-of-order or repeated seq (plan §19)."""
        if seq <= self.last_seq:
            raise ValueError(f"seq {seq} is not after {self.last_seq}")
        self.segments.append(text)
        self.last_seq = seq

    def context_text(self) -> str:
        """Accumulated transcript so far. Head+Tail is applied later by the Predictor."""
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
