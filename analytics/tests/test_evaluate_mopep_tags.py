import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from analytics.src.evaluate_mopep_tags import (
    compare_tags,
    load_dataset,
    parse_model_output,
    resolve_dataset,
    run_evaluation,
    summarize,
)


class EvaluateMopepTagsTests(unittest.TestCase):
    def test_parses_and_compares_model_tags(self):
        generated, parse_ok = parse_model_output("['b2b', 'Serviço']")
        metrics = compare_tags(["b2b", "produto"], generated)

        self.assertTrue(parse_ok)
        self.assertEqual(generated, ["b2b", "serviço"])
        self.assertEqual(metrics["correct_tags"], ["b2b"])
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_loads_dataset_and_generates_missing_example_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.csv"
            path.write_text(
                'business_model,tags\n"Empresa de software","[\'b2b\', \'serviço\']"\n',
                encoding="utf-8",
            )

            dataset = load_dataset(path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.loc[0, "expected_tags"], ["b2b", "serviço"])
            self.assertEqual(len(dataset.loc[0, "example_id"]), 64)
            self.assertEqual(resolve_dataset("calibration", Path(temporary)), path.resolve())

    def test_summarizes_same_core_metrics_as_notebook(self):
        predictions = pd.DataFrame([
            {
                "model": "model:one",
                "expected_tags": ["b2b", "serviço"],
                "generated_tags": ["b2b"],
                "request_ok": True,
                "parse_ok": True,
                "f1": 2 / 3,
                "exact_match": False,
                "latency_ms": 100.0,
            },
            {
                "model": "model:one",
                "expected_tags": ["produto"],
                "generated_tags": ["produto"],
                "request_ok": True,
                "parse_ok": True,
                "f1": 1.0,
                "exact_match": True,
                "latency_ms": 200.0,
            },
        ])

        summary = summarize(predictions).iloc[0]

        self.assertEqual(summary["requests"], 2)
        self.assertAlmostEqual(summary["mean_individual_f1"], 5 / 6)
        self.assertAlmostEqual(summary["exact_match_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_latency_ms"], 150.0)

    def test_run_writes_the_three_result_files(self):
        class FakeClient:
            def list(self):
                return SimpleNamespace(models=[SimpleNamespace(model="model:one")])

            def chat(self, **_kwargs):
                return SimpleNamespace(
                    message=SimpleNamespace(content="['b2b', 'serviço']")
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "calibration.csv"
            dataset.write_text(
                'example_id,business_model,tags\n'
                'one,"Empresa de software","[\'b2b\', \'serviço\']"\n',
                encoding="utf-8",
            )

            run_dir = run_evaluation(
                client=FakeClient(),
                model="model:one",
                dataset_path=dataset,
                output_root=root / "results",
                limit=None,
                seed=42,
                ollama_host="http://127.0.0.1:18000",
            )

            self.assertTrue((run_dir / "predictions.csv").is_file())
            self.assertTrue((run_dir / "performance.csv").is_file())
            self.assertTrue((run_dir / "run.json").is_file())


if __name__ == "__main__":
    unittest.main()
