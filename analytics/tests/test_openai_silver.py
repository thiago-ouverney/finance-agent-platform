import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics.src.openai_silver import (
    BatchPricing,
    build_batch_records,
    enforce_budget,
    estimate_batch_cost,
    parse_batch_output,
    response_body,
    write_jsonl,
    write_run,
)


class OpenAISilverTests(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.DataFrame([{
            "example_id": "company-one",
            "business_model": "Empresa de software B2B.",
            "expected_tags": ["b2b", "serviço"],
            "split": "calibration",
        }])

    def test_batch_request_uses_responses_schema_without_expected_labels(self):
        records, manifest = build_batch_records(self.dataset, "calibration")

        self.assertEqual(records[0]["url"], "/v1/responses")
        self.assertEqual(records[0]["body"]["model"], "gpt-5-nano")
        self.assertEqual(
            records[0]["body"]["text"]["format"]["type"], "json_schema"
        )
        self.assertNotIn("expected_tags", json.dumps(records[0], ensure_ascii=False))
        changed = self.dataset.copy()
        changed.at[0, "expected_tags"] = ["produto"]
        changed_records, _ = build_batch_records(changed, "calibration")
        self.assertEqual(records[0]["body"], changed_records[0]["body"])
        self.assertEqual(manifest.loc[0, "expected_tags"], ["b2b", "serviço"])

    def test_cost_guard_blocks_estimate_above_budget(self):
        records, _ = build_batch_records(self.dataset, "calibration")
        estimate = estimate_batch_cost(records)

        self.assertGreater(estimate["estimated_total_usd"], 0)
        self.assertGreater(estimate["max_total_usd"], estimate["estimated_total_usd"])
        with self.assertRaisesRegex(ValueError, "excede o orçamento"):
            enforce_budget([estimate], max_total_usd=0.000001)

    def test_parses_responses_batch_by_custom_id_and_computes_cost(self):
        records, manifest = build_batch_records(self.dataset, "calibration")
        output = json.dumps({
            "custom_id": records[0]["custom_id"],
            "response": {
                "status_code": 200,
                "body": {
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": json.dumps({"tags": ["b2b", "serviço"]}),
                        }],
                    }],
                    "usage": {
                        "input_tokens": 1000,
                        "input_tokens_details": {"cached_tokens": 200},
                        "output_tokens": 100,
                    },
                },
            },
            "error": None,
        })

        predictions = parse_batch_output(
            output, manifest, "gpt-5-nano", BatchPricing()
        )

        self.assertTrue(predictions.loc[0, "request_ok"])
        self.assertTrue(predictions.loc[0, "parse_ok"])
        self.assertTrue(predictions.loc[0, "exact_match"])
        self.assertAlmostEqual(predictions.loc[0, "cost_usd"], 0.000081)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = write_run(
                predictions=predictions,
                dataset_path=root / "calibration.csv",
                split="calibration",
                model="gpt-5-nano",
                output_root=root / "results",
                batch_id="batch_test",
            )
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

            self.assertTrue((run_dir / "predictions.csv").is_file())
            self.assertTrue((run_dir / "performance.csv").is_file())
            self.assertEqual(metadata["batch_id"], "batch_test")
            self.assertAlmostEqual(metadata["cost_usd"], 0.000081)

    def test_writes_one_json_object_per_line(self):
        records, _ = build_batch_records(self.dataset, "calibration")
        with tempfile.TemporaryDirectory() as temporary:
            path = write_jsonl(records, Path(temporary) / "batch.jsonl")
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["custom_id"], records[0]["custom_id"])

    def test_rejects_invalid_output_limit(self):
        with self.assertRaisesRegex(ValueError, "maior que zero"):
            response_body("texto", max_output_tokens=0)


if __name__ == "__main__":
    unittest.main()
