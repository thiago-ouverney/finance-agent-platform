import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from analytics.src.generate_mopep_silver import (
    load_source_dataset,
    parse_structured_output,
    predict_one,
    run_generation,
)


class FakeCompletions:
    def __init__(self, contents: list[str] | None = None) -> None:
        self.contents = contents or ['{"tags": ["b2b", "serviço"]}']
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.contents[index])
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
            ),
        )


class FakeClient:
    def __init__(
        self,
        *,
        models: list[str] | None = None,
        contents: list[str] | None = None,
    ) -> None:
        model_ids = models if models is not None else ["qwen-silver"]
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=model_id) for model_id in model_ids]
            )
        )
        self.completions = FakeCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


class GenerateMopepSilverTests(unittest.TestCase):
    def test_loads_unlabeled_dataset_and_generates_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "source.csv"
            pd.DataFrame(
                {"business_model": ["Venda software para empresas.", "Loja local."]}
            ).to_csv(dataset_path, index=False)

            first = load_source_dataset(dataset_path)
            second = load_source_dataset(dataset_path)

            self.assertNotIn("expected_tags", first)
            self.assertEqual(first["example_id"].tolist(), second["example_id"].tolist())
            self.assertTrue(first["example_id"].str.fullmatch(r"[0-9a-f]{64}").all())

    def test_structured_output_rejects_unknown_tags(self) -> None:
        tags, ok, error = parse_structured_output('{"tags": ["tag inventada"]}')

        self.assertEqual(tags, [])
        self.assertFalse(ok)
        self.assertIn("fora da taxonomia", error)

    def test_prediction_does_not_send_expected_labels(self) -> None:
        client = FakeClient()
        example = pd.Series(
            {
                "example_id": "example-1",
                "input_sha256": "hash",
                "business_model": "Texto sem o rotulo reservado.",
                "expected_tags": ["rotulo-secreto-fora-da-taxonomia"],
            }
        )

        result = predict_one(
            client,
            model="qwen-silver",
            example=example,
            seed=42,
            max_output_tokens=128,
            max_attempts=1,
            retry_delay_seconds=0,
        )

        request = client.completions.calls[0]
        messages = json.dumps(request["messages"], ensure_ascii=False)
        self.assertNotIn("rotulo-secreto", messages)
        self.assertEqual(result["generated_tags"], ["b2b", "serviço"])
        self.assertTrue(result["parse_ok"])
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["seed"], 42)

    def test_run_is_resumable_without_repeating_completed_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "source.csv"
            output_root = root / "results"
            pd.DataFrame(
                {
                    "example_id": ["a", "b"],
                    "business_model": [
                        "Empresa vende software B2B.",
                        "Comercio local para consumidores.",
                    ],
                }
            ).to_csv(dataset_path, index=False)

            first_client = FakeClient()
            run_dir, first_summary = run_generation(
                first_client,
                dataset_path=dataset_path,
                output_root=output_root,
                base_url="http://127.0.0.1:18000/v1",
                model="qwen-silver",
                model_revision="revision-1",
                limit=None,
                seed=42,
                max_output_tokens=128,
                max_attempts=1,
                retry_delay_seconds=0,
            )

            second_client = FakeClient()
            resumed_dir, second_summary = run_generation(
                second_client,
                dataset_path=dataset_path,
                output_root=output_root,
                base_url="http://127.0.0.1:18000/v1",
                model="qwen-silver",
                model_revision="revision-1",
                limit=None,
                seed=42,
                max_output_tokens=128,
                max_attempts=1,
                retry_delay_seconds=0,
            )

            self.assertEqual(run_dir, resumed_dir)
            self.assertEqual(len(first_client.completions.calls), 2)
            self.assertEqual(len(second_client.completions.calls), 0)
            self.assertEqual(first_summary["pending"], 0)
            self.assertEqual(second_summary["pending"], 0)
            self.assertEqual(len(pd.read_csv(run_dir / "predictions.csv")), 2)
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")

    def test_run_rejects_missing_served_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "source.csv"
            pd.DataFrame({"business_model": ["Empresa de software."]}).to_csv(
                dataset_path, index=False
            )

            with self.assertRaisesRegex(ValueError, "nao encontrado"):
                run_generation(
                    FakeClient(models=["outro-modelo"]),
                    dataset_path=dataset_path,
                    output_root=Path(directory) / "results",
                    base_url="http://127.0.0.1:18000/v1",
                    model="qwen-silver",
                    model_revision="",
                    limit=1,
                    seed=42,
                    max_output_tokens=128,
                    max_attempts=1,
                    retry_delay_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
