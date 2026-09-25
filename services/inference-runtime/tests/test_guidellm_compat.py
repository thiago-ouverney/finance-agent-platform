import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guidellm.scheduler.worker_group import WorkerProcessGroup

from guidellm_compat import completion_drain


class _Event:
    def is_set(self):
        return True


class _State:
    def __init__(self, created, processed, processing):
        self.created_requests = created
        self.processed_requests = processed
        self.processing_requests = processing


class _DelayedMessaging:
    def __init__(self, update=None, delay=0):
        self.update = update
        self.delay = delay

    async def get(self, timeout=None):
        if self.update is None:
            await asyncio.sleep(timeout)
            raise asyncio.TimeoutError
        await asyncio.sleep(self.delay)
        update, self.update = self.update, None
        return update


def _update(state, name):
    return (None, name, None, state)


async def _premature_request_updates(self):
    yield _update(_State(3, 2, 1), "second")


class CompletionDrainTests(unittest.TestCase):
    def test_drains_delayed_real_final_update(self):
        final = _update(_State(3, 3, 0), "third")
        group = object.__new__(WorkerProcessGroup)
        group.shutdown_event = _Event()
        group.messaging = _DelayedMessaging(final, delay=0.01)

        async def collect():
            return [item async for item in group.request_updates()]

        with patch("guidellm_compat.version", return_value="0.7.4"):
            with completion_drain(timeout=0.2):
                updates = asyncio.run(collect())

        self.assertEqual([item[1] for item in updates], ["second", "third"])
        self.assertIs(WorkerProcessGroup.request_updates, _premature_request_updates)

    def test_timeout_preserves_incomplete_result_for_external_guard(self):
        group = object.__new__(WorkerProcessGroup)
        group.shutdown_event = _Event()
        group.messaging = _DelayedMessaging()

        async def collect():
            return [item async for item in group.request_updates()]

        with patch("guidellm_compat.version", return_value="0.7.4"):
            with completion_drain(timeout=0.01):
                updates = asyncio.run(collect())

        self.assertEqual(len(updates), 1)
        state = updates[-1][3]
        self.assertLess(state.processed_requests, state.created_requests)

    @classmethod
    def setUpClass(cls):
        cls.original = WorkerProcessGroup.request_updates
        WorkerProcessGroup.request_updates = _premature_request_updates

    @classmethod
    def tearDownClass(cls):
        WorkerProcessGroup.request_updates = cls.original


if __name__ == "__main__":
    unittest.main()
