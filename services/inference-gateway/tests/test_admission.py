import threading
import time
import unittest

from gateway.admission import AdmissionController, AdmissionRejected


class AdmissionTests(unittest.TestCase):
    def test_allows_up_to_max_inflight_without_queueing(self):
        gate = AdmissionController(max_inflight=8, max_queue=8, retry_after_s=2)
        tokens = [gate.acquire(agent_id=f"a{i}") for i in range(8)]
        self.assertEqual(gate.inflight, 8)
        self.assertEqual(gate.queued, 0)
        for token in tokens:
            token.release()
        self.assertEqual(gate.inflight, 0)

    def test_ninth_request_waits_in_queue_until_a_slot_frees(self):
        gate = AdmissionController(max_inflight=2, max_queue=2, retry_after_s=2)
        first = gate.acquire(agent_id="a1")
        second = gate.acquire(agent_id="a2")
        released = threading.Event()
        result = {}

        def waiter():
            token = gate.acquire(agent_id="a3")
            result["after_release"] = released.is_set()
            token.release()

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        self.assertEqual(gate.queued, 1)
        released.set()
        first.release()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["after_release"])
        second.release()

    def test_rejects_when_queue_is_full(self):
        gate = AdmissionController(max_inflight=1, max_queue=1, retry_after_s=2)
        held = gate.acquire(agent_id="a1")
        queued = {}

        def occupy_queue():
            try:
                queued["token"] = gate.acquire(agent_id="a2")
            except AdmissionRejected as exc:
                queued["error"] = exc

        waiter = threading.Thread(target=occupy_queue)
        waiter.start()
        time.sleep(0.05)
        self.assertEqual(gate.queued, 1)
        with self.assertRaises(AdmissionRejected) as ctx:
            gate.acquire(agent_id="a3")
        self.assertEqual(ctx.exception.retry_after_s, 2)
        self.assertEqual(ctx.exception.status_code, 429)
        held.release()
        waiter.join(timeout=1)
        if "token" in queued:
            queued["token"].release()

    def test_release_serves_the_other_waiting_agent_first(self):
        gate = AdmissionController(max_inflight=1, max_queue=2, retry_after_s=2)
        held = gate.acquire(agent_id="A")
        order: list[str] = []

        def take(agent_id: str) -> None:
            token = gate.acquire(agent_id=agent_id)
            order.append(agent_id)
            token.release()

        same_agent = threading.Thread(target=take, args=("A",))
        other_agent = threading.Thread(target=take, args=("B",))
        same_agent.start()
        self._wait_until(lambda: gate.queued == 1)
        other_agent.start()
        self._wait_until(lambda: gate.queued == 2)
        held.release()
        same_agent.join(timeout=1)
        other_agent.join(timeout=1)
        self.assertEqual(order, ["B", "A"])

    def test_queue_wait_expires_and_leaves_the_line(self):
        gate = AdmissionController(
            max_inflight=1, max_queue=1, retry_after_s=2, queue_timeout_s=0.05,
        )
        held = gate.acquire(agent_id="A")
        result: dict = {}

        def waiter() -> None:
            try:
                gate.acquire(agent_id="B")
            except AdmissionRejected as exc:
                result["error"] = exc

        thread = threading.Thread(target=waiter)
        thread.start()
        self._wait_until(lambda: gate.queued == 1)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["error"].status_code, 503)
        self.assertEqual(result["error"].error_type, "queue_timeout")
        self.assertEqual(gate.queued, 0)
        self.assertEqual(gate.inflight, 1)
        held.release()
        self.assertEqual(gate.inflight, 0)

    def _wait_until(self, predicate) -> None:
        for _ in range(50):
            if predicate():
                return
            time.sleep(0.02)
        self.fail("a fila não chegou ao estado esperado")


if __name__ == "__main__":
    unittest.main()
