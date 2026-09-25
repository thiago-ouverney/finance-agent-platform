"""Narrow compatibility workarounds for the pinned GuideLLM release."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
import time
from typing import Any


GUIDELLM_VERSION = "0.7.4"


def _is_incomplete(state: Any) -> bool:
    """Return whether a scheduler snapshot still has non-terminal requests."""
    return (
        state is not None
        and state.created_requests > state.processed_requests
        and state.processing_requests > 0
    )


@contextmanager
def completion_drain(timeout: float = 5.0) -> Iterator[None]:
    """Drain a late final scheduler update affected by GuideLLM 0.7.4's race.

    GuideLLM 0.7.4 may set ``shutdown_event`` in its receive callback immediately
    before that callback's final update reaches the local receive buffer.  The
    stock iterator can observe the event during its timeout and return first.

    This context manager temporarily wraps ``WorkerProcessGroup.request_updates``.
    After the stock iterator returns, the wrapper reads only genuine messages from
    GuideLLM's receive queue for at most ``timeout`` seconds, and only if the last
    yielded snapshot is demonstrably incomplete and shutdown has been signalled.
    If no update arrives, it returns the incomplete result unchanged so the
    caller's existing completeness guard remains authoritative.

    This is a process-global monkey patch; do not overlap benchmark runs or use the
    context manager concurrently from multiple threads.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    installed = version("guidellm")
    if installed != GUIDELLM_VERSION:
        raise RuntimeError(
            f"completion_drain supports guidellm=={GUIDELLM_VERSION}, found {installed}"
        )

    from guidellm.scheduler.worker_group import WorkerProcessGroup

    original = WorkerProcessGroup.request_updates

    async def request_updates_with_completion_drain(self):
        last_state = None
        async for update in original(self):
            last_state = update[3]
            yield update

        shutdown_event = self.shutdown_event
        messaging = self.messaging
        if (
            not _is_incomplete(last_state)
            or shutdown_event is None
            or not shutdown_event.is_set()
            or messaging is None
        ):
            return

        deadline = time.monotonic() + timeout
        while _is_incomplete(last_state):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                update = await messaging.get(timeout=remaining)
            except asyncio.TimeoutError:
                return
            last_state = update[3]
            yield update

    WorkerProcessGroup.request_updates = request_updates_with_completion_drain
    try:
        yield
    finally:
        WorkerProcessGroup.request_updates = original
