import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import concurrent_load


class ConcurrentLoadTests(unittest.TestCase):
    def test_cycle_repeats_the_frozen_order(self):
        records = ["a", "b", "c"]
        self.assertEqual(
            concurrent_load.cycle_records(records, 8),
            ["a", "b", "c", "a", "b", "c", "a", "b"],
        )

    def test_cycle_rejects_an_empty_source_or_a_non_positive_count(self):
        with self.assertRaises(ValueError):
            concurrent_load.cycle_records([], 1)
        with self.assertRaises(ValueError):
            concurrent_load.cycle_records(["a"], 0)

    def test_closed_loop_keeps_only_the_requested_number_in_flight(self):
        release = threading.Event()
        started = threading.Event()
        lock = threading.Lock()
        state = {"inflight": 0, "max": 0, "entered": 0}

        def request(record, index):
            del record, index
            with lock:
                state["inflight"] += 1
                state["entered"] += 1
                state["max"] = max(state["max"], state["inflight"])
                if state["entered"] == 2:
                    started.set()
            self.assertTrue(release.wait(2))
            with lock:
                state["inflight"] -= 1
            return "ok"

        worker = threading.Thread(
            target=lambda: concurrent_load.run_closed_loop(["a", "b", "c", "d"], request, 2)
        )
        worker.start()
        self.assertTrue(started.wait(2))
        with lock:
            self.assertEqual(state["entered"], 2)
            self.assertEqual(state["max"], 2)
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(state["max"], 2)
        self.assertEqual(state["entered"], 4)

    def test_closed_loop_records_an_error_and_finishes_the_others(self):
        def request(record, index):
            del index
            if record == "bad":
                raise RuntimeError("falhou")
            return record

        report = concurrent_load.run_closed_loop(["ok", "bad", "ok"], request, 2)
        self.assertEqual(report["successful"], ["ok", None, "ok"])
        self.assertIsInstance(report["errored"][1], RuntimeError)
        self.assertEqual(report["successful_count"], 2)
        self.assertEqual(report["errored_count"], 1)

    def test_load_jsonl_requires_messages_and_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.jsonl"
            path.write_text(
                "\n".join([
                    "",
                    json.dumps({"request_id": "a", "initial_messages": [{"role": "user", "content": "oi"}]}),
                    json.dumps({"request_id": "b"}),
                ]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                concurrent_load.load_jsonl(path)
            path.write_text(
                json.dumps({"request_id": "a", "bucket": "short", "initial_messages": [{"role": "user", "content": "oi"}]}) + "\n",
                encoding="utf-8",
            )
            records = concurrent_load.load_jsonl(path)
            self.assertEqual(records[0]["request_id"], "a")

    def test_collect_scalars_reads_nested_context(self):
        payload = {"default_generation_settings": {"n_ctx": 8192}, "total_slots": 1}
        self.assertEqual(concurrent_load.collect_scalars(payload, "n_ctx"), [8192])
        self.assertEqual(concurrent_load.collect_scalars(payload, "total_slots"), [1])
        self.assertEqual(concurrent_load.collect_scalars(payload, "ausente"), [])

    def test_frozen_curve_shape(self):
        self.assertEqual(concurrent_load.STREAMS, (1, 2, 4, 6, 8, 12, 16, 24, 32))
        self.assertEqual(concurrent_load.REQUESTS_PER_REPETITION, 128)
        self.assertEqual(concurrent_load.REPETITIONS, 3)
        self.assertEqual(
            [plan["ctx_size"] // plan["parallel"] for plan in concurrent_load.SLOT_PLANS],
            [8192, 8192, 8192],
        )


if __name__ == "__main__":
    unittest.main()
