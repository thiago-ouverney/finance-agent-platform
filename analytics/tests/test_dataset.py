import csv
import tempfile
import unittest
from pathlib import Path

from analytics.src.build_dataset import collect_rows, write_dataset
from analytics.src.generate_tags import quantization_tag, tag_rows


class AnalyticsTests(unittest.TestCase):
    def test_collects_nested_runtime_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "ollama" / "run-1" / "csv" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text("runtime,model,scenario\nollama,qwen-q4,short\n", encoding="utf-8")
            rows = collect_rows([root])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], "run-1")
            output = root / "dataset.csv"
            write_dataset(rows, output)
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["runtime"], "ollama")

    def test_generates_transparent_bmc_tags(self):
        rows = tag_rows([{
            "runtime": "Ollama",
            "model": "Qwen2.5-7B-Q4_K_M",
            "scenario": "short",
            "phase": "measure",
            "errored_request_count": "0",
            "incomplete_request_count": "0",
            "time_to_first_token_milliseconds_p50": "450",
        }])
        tags = rows[0]["bmc_tags"]
        self.assertIn("runtime:ollama", tags)
        self.assertIn("quantization:q4", tags)
        self.assertIn("performance:ttft-fast", tags)
        self.assertEqual(quantization_tag("model-AWQ"), "awq")


if __name__ == "__main__":
    unittest.main()
