from __future__ import annotations

import threading
from dataclasses import dataclass


class AdmissionRejected(Exception):
    def __init__(
        self,
        retry_after_s: int,
        status_code: int = 429,
        error_type: str = "admission_rejected",
    ) -> None:
        message = (
            "queue wait timed out"
            if error_type == "queue_timeout"
            else f"admission rejected; retry after {retry_after_s}s"
        )
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.status_code = status_code
        self.error_type = error_type


@dataclass
class AdmissionToken:
    _gate: AdmissionController

    def release(self) -> None:
        self._gate.release()


@dataclass
class _Waiter:
    agent_id: str
    ready: threading.Event


class AdmissionController:
    """Limits concurrent upstream calls and rotates the queue across agents."""

    def __init__(
        self,
        max_inflight: int,
        max_queue: int,
        retry_after_s: int,
        queue_timeout_s: float = 8,
    ) -> None:
        if max_inflight < 1 or max_queue < 0 or retry_after_s < 0 or queue_timeout_s <= 0:
            raise ValueError("admission limits must be non-negative; inflight >= 1; timeout > 0")
        self.max_inflight = max_inflight
        self.max_queue = max_queue
        self.retry_after_s = retry_after_s
        self.queue_timeout_s = queue_timeout_s
        self._lock = threading.Lock()
        self._inflight = 0
        self._waiters: list[_Waiter] = []
        self._last_agent: str | None = None

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def queued(self) -> int:
        with self._lock:
            return len(self._waiters)

    def acquire(self, agent_id: str) -> AdmissionToken:
        with self._lock:
            if self._inflight < self.max_inflight and not self._waiters:
                self._grant(agent_id)
                return AdmissionToken(self)
            if len(self._waiters) >= self.max_queue:
                raise AdmissionRejected(self.retry_after_s)
            waiter = _Waiter(agent_id, threading.Event())
            self._waiters.append(waiter)
        if waiter.ready.wait(self.queue_timeout_s):
            return AdmissionToken(self)
        with self._lock:
            if waiter not in self._waiters:
                return AdmissionToken(self)
            self._waiters.remove(waiter)
        raise AdmissionRejected(self.retry_after_s, status_code=503, error_type="queue_timeout")

    def release(self) -> None:
        with self._lock:
            if self._inflight <= 0:
                raise RuntimeError("release without matching acquire")
            self._inflight -= 1
            if not self._waiters:
                return
            index = next(
                (i for i, waiter in enumerate(self._waiters) if waiter.agent_id != self._last_agent),
                0,
            )
            waiter = self._waiters.pop(index)
            self._grant(waiter.agent_id)
            waiter.ready.set()

    def _grant(self, agent_id: str) -> None:
        self._inflight += 1
        self._last_agent = agent_id
