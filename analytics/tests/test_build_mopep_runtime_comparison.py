import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics.src.build_mopep_runtime_comparison import build


class BuildMopepRuntimeComparisonTests(unittest.TestCase):
    def test_scores_quality_and_review_delta_without_sending_golden_to_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            responses_dir = root / "results" / "llama" / "run" / "text"
            responses_dir.mkdir(parents=True)
            rows = [
                {"experiment_id": "closed", "runtime": "llama", "benchmark_phase": "measure", "status": "successful", "workload_request_id": "one", "workload_profile": "mopep-review-closed-loop", "bucket": "short", "turn_index": 1, "repetition": 1, "assistant_output": "['produto']"},
                {"experiment_id": "closed", "runtime": "llama", "benchmark_phase": "measure", "status": "successful", "workload_request_id": "one", "workload_profile": "mopep-review-closed-loop", "bucket": "short", "turn_index": 2, "repetition": 1, "assistant_output": "['b2b']"},
            ]
            (responses_dir / "responses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            json_dir = responses_dir.parent / "json"
            json_dir.mkdir()
            (json_dir / "manifest.json").write_text(json.dumps({"status": "complete"}))
            response_sha = hashlib.sha256((responses_dir / "responses.jsonl").read_bytes()).hexdigest()
            (json_dir / "responses-manifest.json").write_text(json.dumps({
                "responses_sha256": response_sha,
            }))
            golden = root / "golden.csv"
            golden.write_text('id,tags\none,"[\'b2b\']"\n', encoding="utf-8")
            observability = root / "observability"
            observability.mkdir()
            pd.DataFrame([{
                "experiment_id": "closed", "workload_request_id": "one", "turn_index": 2,
                "benchmark_phase": "measure", "comparison_id": "same", "repetition": 1,
                "time_to_first_token_ms": 10,
            }]).to_csv(observability / "all-requests.csv", index=False)
            output = root / "output"

            manifest = build(
                results_dir=root / "results", observability_dir=observability,
                golden_path=golden, output_dir=output,
            )

            deltas = pd.read_csv(output / "review-deltas.csv")
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(deltas.loc[0, "f1_delta"], 1.0)
            self.assertTrue(deltas.loc[0, "improved"])
            self.assertTrue((output / "quality-summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
