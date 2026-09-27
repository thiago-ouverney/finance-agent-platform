import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics.src.consolidate_mopep_evals import (
    collect_runs,
    sample_sha256,
    write_consolidated,
)


class ConsolidateMopepEvalsTests(unittest.TestCase):
    def write_run(self, root: Path, run_id: str, model: str, model_info=None) -> None:
        run_dir = root / run_id
        run_dir.mkdir()
        metadata = {
            "run_id": run_id,
            "created_at": "2026-09-26T12:00:00+00:00",
            "model": model,
            "model_info": model_info or {},
            "dataset": "/tmp/mopep-bmc-tags/calibration.csv",
            "dataset_rows": 1,
            "seed": 42,
            "limit": 1,
            "prompt_sha256": "prompt-one",
            "ollama_options": {"temperature": 0.0, "num_predict": 512, "seed": 42},
        }
        (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
        (run_dir / "performance.csv").write_text(
            "model,requests,request_success_rate,parse_rate,mean_individual_f1,"
            "exact_match_rate,mean_latency_ms,macro_f1\n"
            f"{model},1,1.0,1.0,0.8,0.0,1000.0,0.75\n",
            encoding="utf-8",
        )
        (run_dir / "predictions.csv").write_text(
            "example_id,input_sha256,model,expected_tags,f1,latency_ms,parse_ok\n"
            f"example-one,input-one,{model},\"['b2b']\",0.8,1000.0,True\n",
            encoding="utf-8",
        )

    def test_collects_runs_and_predictions_for_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_run(
                root,
                "run-one",
                "qwen2.5:7b-instruct-q4_K_M",
                {
                    "digest": "abcdef1234567890",
                    "size_bytes": 4_700_000_000,
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                },
            )
            self.write_run(root, "run-two", "qwen2.5:14b-instruct-q8_0")

            runs, predictions = collect_runs(root)

            self.assertEqual(len(runs), 2)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(runs["comparison_id"].nunique(), 1)
            self.assertEqual(runs.loc[0, "model_size_gb"], 4.7)
            self.assertEqual(runs.loc[1, "parameter_size_b"], 14.0)
            self.assertEqual(runs.loc[1, "quantization_level"].lower(), "q8_0")
            self.assertIn("abcdef12", runs.loc[0, "model_label"])

            runs_path, predictions_path = write_consolidated(
                runs, predictions, root / "consolidated"
            )
            self.assertTrue(runs_path.is_file())
            self.assertTrue(predictions_path.is_file())

    def test_sample_signature_changes_with_expected_tags(self):
        first = pd.DataFrame([{"example_id": "one", "expected_tags": '["b2b"]'}])
        second = pd.DataFrame([{"example_id": "one", "expected_tags": '["b2c"]'}])

        self.assertNotEqual(sample_sha256(first), sample_sha256(second))


if __name__ == "__main__":
    unittest.main()
