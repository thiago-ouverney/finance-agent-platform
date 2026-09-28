import json
import tempfile
import unittest
from pathlib import Path

import workload_dataset


def record(request_id="one"):
    return {
        "request_id": request_id, "bucket": "short", "rendered_prompt_tokens": 10,
        "initial_messages": [{"role": "user", "content": "classifique"}],
        "review_instruction": "revise",
    }


class WorkloadDatasetTests(unittest.TestCase):
    def fake_request(self, _cfg, messages, _timeout, _secret, metadata):
        return {**metadata, "output": "['b2b']", "request_sha256": str(len(messages))}

    def test_profiles_use_one_one_and_two_calls(self):
        single = workload_dataset.run_batch(
            cfg={}, records=[record()], profile="mopep-single", timeout=1,
            secret="", request=self.fake_request,
        )
        replay = workload_dataset.run_batch(
            cfg={}, records=[record()], profile="mopep-review-replay", timeout=1,
            secret="", request=self.fake_request, replay={"one": "['produto']"},
        )
        closed = workload_dataset.run_batch(
            cfg={}, records=[record()], profile="mopep-review-closed-loop", timeout=1,
            secret="", request=self.fake_request,
        )
        self.assertEqual(len(single["benchmarks"][0]["requests"]["successful"]), 1)
        self.assertEqual(len(replay["benchmarks"][0]["requests"]["successful"]), 1)
        self.assertEqual(len(closed["benchmarks"][0]["requests"]["successful"]), 2)
        self.assertEqual(replay["benchmarks"][0]["requests"]["successful"][0]["turn_index"], 2)

    def test_rejects_duplicate_ids_and_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = root / "workload.jsonl"
            workload.write_text("\n".join(json.dumps(record()) for _ in range(2)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicado"):
                workload_dataset.load_jsonl(workload)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "workload_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "diverge"):
                workload_dataset.load_workload(workload, manifest)

    def test_replay_requires_llama_manifest_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.jsonl"
            workload_dataset.write_responses(path, [{
                "workload_request_id": "one", "turn_index": 1,
                "assistant_output": "['b2b']",
            }])
            manifest_path = workload_dataset.replay_manifest_path(path)
            manifest_path.write_text(json.dumps({
                "source_runtime": "llama",
                "responses_sha256": workload_dataset.sha256_file(path),
            }))
            mapping, digest, manifest = workload_dataset.load_replay(path)
            self.assertEqual(mapping["one"], "['b2b']")
            self.assertEqual(digest, manifest["responses_sha256"])


if __name__ == "__main__":
    unittest.main()
